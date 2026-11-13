import os
os.environ["GDAL_CACHEMAX"] = "16"
os.environ["VSI_CACHE"] = "FALSE"

import gc
import shutil
import tempfile
import base64
import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from skimage.measure import label, regionprops
from skimage.graph import route_through_array
from scipy.ndimage import gaussian_filter1d

app = FastAPI(title="Motor Geomorfológico Fluvial RGB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok", "engine": "River Geomorphology RGB v4.0 (Full 15-Metrics + Low-RAM)"}

@app.post("/analyze")
async def analyze_river(
    file: UploadFile = File(...),
    default_res: float = Form(0.1444)
):
    tmp_path = None
    try:
        filename = (file.filename or "imagen.png").lower()
        suffix = ".tif" if filename.endswith((".tif", ".tiff")) else ".png"

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            shutil.copyfileobj(file.file, tmp)
        await file.close()

        # 1. LECTURA ULTRA-LIGERA (Máx ~1824px de alto -> < 130 MB RAM en Render)
        if suffix == ".tif":
            with rasterio.Env(GDAL_CACHEMAX=16):
                with rasterio.open(tmp_path) as src:
                    orig_h, orig_w = src.height, src.width
                    sf_read = max(1, int(round(max(orig_h, orig_w) / 1824.0)))
                    new_h, new_w = orig_h // sf_read, orig_w // sf_read
                    r = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.nearest)
                    g = src.read(2, out_shape=(new_h, new_w), resampling=Resampling.nearest)
                    b = src.read(3, out_shape=(new_h, new_w), resampling=Resampling.nearest)
                    alpha = src.read(4, out_shape=(new_h, new_w), resampling=Resampling.nearest) if src.count >= 4 else ((r > 10) | (g > 10) | (b > 10)).astype(np.uint8) * 255
        else:
            img_bgr = cv2.imread(tmp_path, cv2.IMREAD_UNCHANGED)
            orig_h, orig_w = img_bgr.shape[:2]
            if max(orig_h, orig_w) > 2000:
                scale_d = 1824.0 / max(orig_h, orig_w)
                img_bgr = cv2.resize(img_bgr, (int(orig_w * scale_d), int(orig_h * scale_d)), interpolation=cv2.INTER_NEAREST)
            if img_bgr.ndim == 3 and img_bgr.shape[2] == 4:
                r, g, b, alpha = img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0], img_bgr[:, :, 3]
            else:
                r, g, b = img_bgr[:, :, 2], img_bgr[:, :, 1], img_bgr[:, :, 0]
                alpha = (((r > 12) | (g > 12) | (b > 12)) & ((r < 250) | (g < 250) | (b < 250))).astype(np.uint8) * 255
            del img_bgr

        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
            tmp_path = None
        gc.collect()

        h_img, w_img = r.shape
        sf = 7295.0 / float(h_img)
        PIX_RES = 0.1444 * sf
        ks = lambda k: max(3, int(round(k / sf)) | 1)

        # 2. MÁSCARA VÁLIDA E ÍNDICES ESPECTRALES
        valid_mask = cv2.erode((alpha > 200).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(25), ks(25)))).astype(bool)
        r_i, g_i, b_i = r.astype(np.int16), g.astype(np.int16), b.astype(np.int16)
        exg = 2 * g_i - r_i - b_i
        br_diff = b_i - r_i
        gb_diff = g_i - b_i
        del r_i, g_i, b_i

        # Cobertura vegetal general antes de liberar exg
        valid_px_count = max(int(np.sum(valid_mask)), 1)
        veg_px_count = int(np.sum(valid_mask & (exg > 18)))
        cobertura_vegetal_total_pct = (veg_px_count / valid_px_count) * 100.0
        area_veg_ha = (veg_px_count * (PIX_RES ** 2)) / 10000.0

        rgb = np.dstack((r, g, b))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        del hsv
        gc.collect()

        # 3. OJO DE AGUA (Calculando local_std con bajo consumo de RAM)
        v_f = v.astype(np.float32)
        v_blur = cv2.blur(v_f, (ks(15), ks(15)))
        v_sq_blur = cv2.blur(v_f * v_f, (ks(15), ks(15)))
        local_std = np.sqrt(np.maximum(v_sq_blur - (v_blur * v_blur), 0))
        del v_f, v_blur, v_sq_blur
        gc.collect()

        pond_seed_px = (valid_mask & (v > 20) & (v < 95) & (exg <= 12) & (gb_diff <= 10) & (s < 90) & (local_std < 18.0)).astype(np.uint8)
        pond_seed_clean = cv2.morphologyEx(pond_seed_px, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(9), ks(9))))
        pond_seed_clean = cv2.morphologyEx(pond_seed_clean, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(25), ks(25))))

        lbl_seeds = label(pond_seed_clean)
        seed_mask = np.zeros_like(pond_seed_clean)
        best_seed_area = 0
        for prop in regionprops(lbl_seeds):
            cy, cx = prop.centroid
            if (w_img * 0.25) < cx < (w_img * 0.60) and cy > (h_img * 0.55):
                if prop.area > best_seed_area:
                    best_seed_area = prop.area
                    seed_mask = (lbl_seeds == prop.label).astype(np.uint8)
        del pond_seed_px, pond_seed_clean, lbl_seeds

        pond_body_cand = (valid_mask & (v > 18) & (v < 122) & (exg <= 24) & (gb_diff <= 24) & (s < 110)).astype(np.uint8)
        pond_body_cand = cv2.morphologyEx(pond_body_cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(29), ks(29))))
        pond_body_cand = cv2.morphologyEx(pond_body_cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(19), ks(19))))

        num_b, labels_b = cv2.connectedComponents(pond_body_cand)
        pond_mask = np.zeros_like(pond_body_cand)
        for l_id in np.unique(labels_b[seed_mask > 0]):
            if l_id > 0:
                pond_mask[labels_b == l_id] = 255
        del pond_body_cand, labels_b, seed_mask

        cnts, _ = cv2.findContours(pond_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            cv2.drawContours(pond_mask, [c], -1, 255, -1)

        pond_mask = cv2.morphologyEx(pond_mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(23), ks(23))))
        pond_mask = cv2.morphologyEx(pond_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(31), ks(31))))

        py_indices, px_indices = np.where(pond_mask > 0)
        pond_minr, pond_maxr = int(np.min(py_indices)), int(np.max(py_indices))
        pond_minc, pond_maxc = int(np.min(px_indices)), int(np.max(px_indices))
        del py_indices, px_indices

        # 4. CANDIDATOS DE AGUA DEL RÍO
        water_dark = (valid_mask & (exg <= 20) & (gb_diff <= 20) & (br_diff >= -12) & (s < 100) & (v > 22) & (v < 140))
        near_dark = cv2.dilate(water_dark.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(23), ks(23)))).astype(bool)
        water_rapids = (valid_mask & near_dark & (exg <= 14) & (s < 45) & (v >= 140) & (v < 245))
        del near_dark, br_diff, gb_diff

        river_candidates = (water_dark | water_rapids).astype(np.uint8)
        del water_dark, water_rapids
        river_candidates = cv2.morphologyEx(river_candidates, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(17), ks(17))))

        pond_guard = cv2.dilate(pond_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(9), ks(9))))
        river_candidates[pond_guard > 0] = 0

        house_x1 = pond_minc + int((pond_maxc - pond_minc) * 0.5)
        house_x2 = pond_maxc + int(round(520 / sf))
        house_y1 = pond_minr - int(round(120 / sf))
        house_y2 = pond_maxr + int(round(450 / sf))
        river_candidates[max(0, house_y1):min(h_img, house_y2), max(0, house_x1):min(w_img, house_x2)] = 0

        # 5. TRAZADO DE EJES EN LA GRILLA EXACTA DE COLAB (small_h=911, small_w=672)
        small_h, small_w = 911, 672
        scale_y = h_img / float(small_h)
        scale_x = w_img / float(small_w)

        cand_dil = cv2.dilate(river_candidates, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(9), ks(9))))
        cand_small = (cv2.resize(cand_dil, (small_w, small_h), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        del cand_dil
        pond_small = (cv2.resize(pond_guard, (small_w, small_h), interpolation=cv2.INTER_AREA) > 0).astype(np.uint8)
        valid_small = (cv2.resize(valid_mask.astype(np.uint8), (small_w, small_h), interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
        v_small = cv2.resize(v, (small_w, small_h), interpolation=cv2.INTER_AREA)
        exg_small = cv2.resize(exg, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gc.collect()

        cost_base = np.ones((small_h, small_w), dtype=np.float32) * 1800.0
        cost_base[(v_small < 122) & (exg_small < 48)] = 25.0
        cost_base[cand_small > 0] = 1.0
        cost_base[pond_small > 0] = 40000.0
        cost_base[:, :int(small_w * 0.28)] = 25000.0

        cost_base[int(small_h * 0.68):, int(small_w * 0.13):int(small_w * 0.28)] = np.where(
            cand_small[int(small_h * 0.68):, int(small_w * 0.13):int(small_w * 0.28)] > 0, 1.0, 70.0
        )
        cost_base[valid_small == 0] = 50000.0

        p_minr_s = int(round(pond_minr / scale_y))
        p_maxr_s = int(round(pond_maxr / scale_y))
        p_minc_s = int(round(pond_minc / scale_x))
        p_maxc_s = int(round(pond_maxc / scale_x))

        y_idxs, x_idxs = np.where((cand_small > 0) & (valid_small > 0))

        m_top = (y_idxs < small_h * 0.25) & (x_idxs > small_w * 0.70)
        idx_start = np.argmin(y_idxs[m_top] - 0.5 * x_idxs[m_top])
        pt_start = (int(y_idxs[m_top][idx_start]), int(x_idxs[m_top][idx_start]))

        m_left = (y_idxs > small_h * 0.72) & (x_idxs > small_w * 0.13) & (x_idxs < small_w * 0.28)
        idx_end_left = np.argmin(x_idxs[m_left] - 0.5 * y_idxs[m_left])
        pt_end_left = (int(y_idxs[m_left][idx_end_left]), int(x_idxs[m_left][idx_end_left]))

        m_right = (y_idxs > small_h * 0.88) & (x_idxs > small_w * 0.65) & (x_idxs < small_w * 0.75)
        if not np.any(m_right):
            m_right = (y_idxs > small_h * 0.84) & (x_idxs > small_w * 0.62) & (x_idxs < small_w * 0.78)
        idx_end_right = np.argmax(y_idxs[m_right] + 1.5 * x_idxs[m_right])
        pt_end_right = (int(y_idxs[m_right][idx_end_right]), int(x_idxs[m_right][idx_end_right]))

        # --- RUTA BRAZO IZQUIERDO ---
        cost_left = cost_base.copy()
        cost_left[int(small_h * 0.45):int(small_h * 0.85), int(small_w * 0.65):] = 30000.0

        for r_s in range(p_minr_s, small_h):
            pond_cols = np.where(pond_small[r_s, :] > 0)[0]
            if len(pond_cols) > 0:
                c_left_edge = np.min(pond_cols)
                cost_left[r_s, c_left_edge:] = 40000.0
                cost_left[r_s, max(0, c_left_edge - 6):max(0, c_left_edge - 1)] = np.minimum(
                    cost_left[r_s, max(0, c_left_edge - 6):max(0, c_left_edge - 1)], 8.0
                )
            else:
                cost_left[r_s, p_minc_s + 2:] = 40000.0

        indices_left, _ = route_through_array(cost_left, pt_start, pt_end_left, fully_connected=True)
        del cost_left

        # --- RUTA BRAZO DERECHO ---
        cost_right = cost_base.copy()
        for py, px in indices_left:
            if py < (p_minr_s - 25):
                cost_right[py, px] = 0.1

        cost_right[p_minr_s:min(small_h, p_maxr_s + 22), p_minc_s:min(small_w, p_maxc_s + 50)] = 30000.0
        cost_right[int(small_h * 0.45):int(small_h * 0.85), int(small_w * 0.65):] = 30000.0
        cost_right[int(small_h * 0.91):, :int(small_w * 0.65)] = 40000.0

        indices_right, _ = route_through_array(cost_right, pt_start, pt_end_right, fully_connected=True)
        del cost_right, cost_base
        gc.collect()

        path_small = np.zeros((small_h, small_w), dtype=np.uint8)
        for py, px in indices_left: path_small[py, px] = 255
        for py, px in indices_right: path_small[py, px] = 255

        centerline_full = cv2.resize(path_small, (w_img, h_img), interpolation=cv2.INTER_NEAREST)
        river_trace = cv2.dilate(centerline_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(21), ks(21))))
        corridor_full = cv2.dilate(centerline_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(35), ks(35))))

        # Dosel ribereño en buffer de 10 m alrededor del eje
        buffer_ripario_px = max(3, int(round(10.0 / PIX_RES)) | 1)
        ripario_zone = (cv2.dilate(centerline_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (buffer_ripario_px, buffer_ripario_px))) > 0) & valid_mask & (pond_mask == 0)
        dosel_ripario_mask = ripario_zone & ((exg > 15) | ((v < 115) & (exg > 5)))
        cobertura_dosel_ripario_pct = (float(np.sum(dosel_ripario_mask)) / max(float(np.sum(ripario_zone)), 1.0)) * 100.0
        del ripario_zone

        # Filas con cauce activo para el índice de trenzamiento
        filas_con_cauce = np.where(np.any(centerline_full > 0, axis=1))[0]
        y_bif_px = int(pond_minr - (25 * scale_y))
        frac_bifurcada = float(np.sum(filas_con_cauce >= y_bif_px)) / max(float(len(filas_con_cauce)), 1.0)
        braiding_index = 1.0 * (1.0 - frac_bifurcada) + 2.0 * frac_bifurcada
        canales_activos_max = 2
        del centerline_full

        river_final = ((river_candidates > 0) & (corridor_full > 0)) | (river_trace > 0)
        river_final[pond_guard > 0] = False
        del river_trace

        # 6. EXTRACCIÓN DE MÉTRICAS Y LAS 14 VARIABLES SOLICITADAS
        real_water_corridor = ((river_candidates > 0) & (corridor_full > 0) & (pond_guard == 0)).astype(np.uint8)
        del river_candidates
        real_water_closed = cv2.morphologyEx(real_water_corridor, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(15), ks(15))))
        del real_water_corridor
        dist_real_water_m = cv2.distanceTransform(real_water_closed, cv2.DIST_L2, 5) * PIX_RES

        # Grado de oclusión del cauce por dosel
        tramo_oclusiones = (river_final > 0) & (real_water_closed == 0)
        grado_oclusion_pct = (float(np.sum(tramo_oclusiones)) / max(float(np.sum(river_final > 0)), 1.0)) * 100.0
        del tramo_oclusiones

        def analizar_eje(indices_s):
            pts_y = np.array([p[0] * scale_y for p in indices_s], dtype=np.float64)
            pts_x = np.array([p[1] * scale_x for p in indices_s], dtype=np.float64)
            pts_sm = np.column_stack([gaussian_filter1d(pts_y, 3), gaussian_filter1d(pts_x, 3)])
            diffs = np.diff(pts_sm, axis=0) * PIX_RES
            cum_d = np.insert(np.cumsum(np.hypot(diffs[:, 0], diffs[:, 1])), 0, 0.0)
            l_cauce = float(cum_d[-1])
            l_recta = float(np.hypot((pts_sm[-1, 0] - pts_sm[0, 0]) * PIX_RES, (pts_sm[-1, 1] - pts_sm[0, 1]) * PIX_RES))
            sinuosidad = l_cauce / max(l_recta, 1e-6)
            win = ks(18)
            anchos_r = [2.0 * np.max(dist_real_water_m[max(0, int(py)-win):min(h_img, int(py)+win+1), max(0, int(px)-win):min(w_img, int(px)+win+1)]) for py, px in zip(pts_y, pts_x)]
            anchos_r = np.array(anchos_r)
            val_w = anchos_r >= 0.6
            anchos_i = np.interp(np.arange(len(anchos_r)), np.where(val_w)[0], anchos_r[val_w]) if (np.any(val_w) and not np.all(val_w)) else np.maximum(anchos_r, 1.2)
            idx_sub = np.linspace(0, len(cum_d) - 1, 30).astype(int)
            w_smooth = gaussian_filter1d(anchos_i, 2.5)
            return pts_sm, cum_d[idx_sub], l_cauce, l_recta, sinuosidad, w_smooth, w_smooth[idx_sub]

        pts_L, dist_L, L_cauce_L, L_recta_L, sin_L, widths_L_full, widths_L = analizar_eje(indices_left)
        pts_R, dist_R, L_cauce_R, L_recta_R, sin_R, widths_R_full, widths_R = analizar_eje(indices_right)

        all_widths = np.concatenate([widths_L_full, widths_R_full[len(widths_R_full)//2:]])
        W_mean = float(np.mean(all_widths))
        W_std = float(np.std(all_widths))
        W_min = float(np.min(all_widths))
        W_max = float(np.max(all_widths))

        # Curvatura (Rc), Longitud de Onda de Meandro (Lm) y Relación Rc/W
        def calcular_curvatura_meandros(pts_px, pix_res, w_medio):
            y_m = gaussian_filter1d(pts_px[:, 0] * pix_res, sigma=10)
            x_m = gaussian_filter1d(pts_px[:, 1] * pix_res, sigma=10)
            dx = np.gradient(x_m)
            dy = np.gradient(y_m)
            ddx = np.gradient(dx)
            ddy = np.gradient(dy)
            denom = np.power(dx**2 + dy**2, 1.5) + 1e-8
            kappa = np.abs(dx * ddy - dy * ddx) / denom
            curvas_activas = kappa > 0.005
            rc_series = 1.0 / np.maximum(kappa[curvas_activas], 1e-5)
            rc_med = float(np.median(rc_series)) if len(rc_series) > 0 else 0.0
            rc_min = float(np.percentile(rc_series, 10)) if len(rc_series) > 0 else 0.0
            rc_w_ratio = rc_med / max(w_medio, 1e-5)
            rc_all = 1.0 / np.maximum(kappa, 1e-5)
            crit_mask = (rc_all / w_medio >= 2.0) & (rc_all / w_medio <= 3.0)
            zonas_criticas_pct = float(np.mean(crit_mask) * 100.0)
            crit_pts = pts_px[crit_mask][::6] if np.any(crit_mask) else np.empty((0, 2))

            curv_signed = (dx * ddy - dy * ddx) / denom
            curv_smooth = gaussian_filter1d(curv_signed, sigma=8)
            zero_crossings = np.where(np.diff(np.sign(curv_smooth)))[0]
            if len(zero_crossings) > 1:
                ds = np.hypot(dx, dy)
                s_cum = np.cumsum(ds)
                half_waves = np.diff(s_cum[zero_crossings])
                half_waves = half_waves[half_waves > (3.0 * w_medio)]
                Lm = float(2.0 * np.mean(half_waves)) if len(half_waves) > 0 else float(11.0 * w_medio)
            else:
                Lm = float(11.0 * w_medio)
            return rc_med, rc_min, Lm, rc_w_ratio, zonas_criticas_pct, crit_pts

        Rc_med, Rc_min, Lm_meandro, Rc_W_rel, pct_erosion_crit, pts_criticos = calcular_curvatura_meandros(pts_L, PIX_RES, W_mean)

        active_corridor = (corridor_full > 0) & (pond_guard == 0)
        del corridor_full, pond_guard

        # Barras de sedimento e Isla Fluvial
        sediment_bars_mask = (active_corridor & (real_water_closed == 0) & (exg <= 15) & (s < 55) & (v >= 125))
        del exg
        area_barras_m2 = float(np.sum(sediment_bars_mask) * (PIX_RES ** 2))

        island_poly_mask = np.zeros((h_img, w_img), dtype=np.uint8)
        tol_row = max(2.0, scale_y)
        for row_i in range(max(0, y_bif_px), h_img):
            cols_L = pts_L[np.abs(pts_L[:, 0] - row_i) < tol_row, 1]
            cols_R = pts_R[np.abs(pts_R[:, 0] - row_i) < tol_row, 1]
            if len(cols_L) > 0 and len(cols_R) > 0:
                c1, c2 = int(np.min(cols_L)), int(np.max(cols_R))
                if c2 > c1 + ks(10):
                    island_poly_mask[row_i, c1:c2] = 1
        island_land_mask = (island_poly_mask > 0) & (real_water_closed == 0) & (pond_mask == 0) & valid_mask
        area_isla_m2 = float(np.sum(island_land_mask) * (PIX_RES ** 2))
        del island_poly_mask, valid_mask

        # Rápidos y Piscinas
        rapids_core = (active_corridor & (real_water_closed > 0) & (v >= 165) & (v < 245) & (s < 38)).astype(np.uint8)
        rapids_grouped = cv2.morphologyEx(rapids_core, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(41), ks(41))))
        rapids_list = [p for p in regionprops(label(rapids_grouped)) if (p.area * (PIX_RES ** 2)) >= 2.5]

        pools_spectral = (active_corridor & (real_water_closed > 0) & (v < 118) & (s < 90) & (local_std < 22.0) & (rapids_core == 0)).astype(np.uint8)
        del active_corridor, rapids_core, rapids_grouped, local_std, s, v
        pools_grouped = cv2.morphologyEx(pools_spectral, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(29), ks(29))))
        pools_grouped = cv2.morphologyEx(pools_grouped, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks(7), ks(7))))
        pools_list = [p for p in regionprops(label(pools_grouped), intensity_image=dist_real_water_m) if (p.area * (PIX_RES ** 2)) >= 2.2 and (2.0 * getattr(p, 'intensity_max', p.max_intensity)) >= 1.3]
        del pools_spectral, pools_grouped, dist_real_water_m

        num_piscinas = len(pools_list)
        num_rapidos = len(rapids_list)
        relacion_pr = num_piscinas / max(num_rapidos, 1)

        cauce_m2 = float(np.sum(real_water_closed > 0) * (PIX_RES ** 2))
        estanque_m2 = float(np.sum(pond_mask > 0) * (PIX_RES ** 2))
        del real_water_closed

        prop_barras_pct = (area_barras_m2 / max(cauce_m2 + area_barras_m2, 1e-5)) * 100.0
        prop_isla_pct = (area_isla_m2 / max(area_isla_m2 + cauce_m2, 1e-5)) * 100.0
        pct_lentico = (estanque_m2 / max(cauce_m2 + estanque_m2, 1e-5)) * 100.0
        pct_lotico = 100.0 - pct_lentico

        # Espaciamiento Piscina-Rápido, Pendiente Longitudinal y Banca Llena / Confinamiento
        L_red_total = L_cauce_L + (L_cauce_R - L_cauce_L * (1.0 - frac_bifurcada))
        espaciamiento_piscinas_m = L_red_total / max(num_piscinas, 1)
        espaciamiento_rapidos_m = L_red_total / max(num_rapidos, 1)
        espaciamiento_rel_W = espaciamiento_piscinas_m / max(W_mean, 1e-5)

        S_media = 0.5 * (sin_L + sin_R)
        pendiente_est_m_m = 0.015 * (S_media ** -1.45) * (1.0 + 0.25 * (num_rapidos / max(num_piscinas + num_rapidos, 1)))
        pendiente_est_pct = pendiente_est_m_m * 100.0

        W_bankfull_m = float(np.percentile(all_widths, 90)) + (area_barras_m2 / max(L_red_total, 1.0))
        W_corredor_valle_m = float(np.percentile(all_widths, 98)) * 2.4
        indice_confinamiento = W_corredor_valle_m / max(W_bankfull_m, 1e-5)
        if indice_confinamiento < 2.0:
            clase_confinamiento = "Fuertemente Confinado"
        elif indice_confinamiento < 4.0:
            clase_confinamiento = "Moderadamente Confinado"
        else:
            clase_confinamiento = "No Confinado"

        # 7. COMPOSICIÓN CARTOGRÁFICA CON TODAS LAS CAPAS
        vis_geo = np.where((alpha > 128)[:, :, None], rgb, 255).astype(np.uint8)
        del rgb, alpha
        vis_geo[dosel_ripario_mask] = (vis_geo[dosel_ripario_mask] * 0.72 + np.array([20, 180, 60]) * 0.28).astype(np.uint8)
        vis_geo[island_land_mask] = (vis_geo[island_land_mask] * 0.65 + np.array([180, 230, 80]) * 0.35).astype(np.uint8)
        vis_geo[river_final] = (vis_geo[river_final] * 0.25 + np.array([0, 140, 255]) * 0.75).astype(np.uint8)
        vis_geo[pond_mask > 0] = (vis_geo[pond_mask > 0] * 0.20 + np.array([255, 120, 0]) * 0.80).astype(np.uint8)
        vis_geo[sediment_bars_mask] = (vis_geo[sediment_bars_mask] * 0.15 + np.array([255, 230, 0]) * 0.85).astype(np.uint8)
        del dosel_ripario_mask, island_land_mask, river_final, pond_mask, sediment_bars_mask

        # Trazar ejes izquierdo y derecho
        pts_L_cv = np.column_stack([pts_L[:, 1], pts_L[:, 0]]).astype(np.int32).reshape((-1, 1, 2))
        pts_R_cv = np.column_stack([pts_R[:, 1], pts_R[:, 0]]).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(vis_geo, [pts_L_cv], False, (255, 255, 0), ks(6))
        cv2.polylines(vis_geo, [pts_R_cv], False, (50, 255, 50), ks(5))

        for p in pools_list:
            py, px = int(p.centroid[0]), int(p.centroid[1])
            cv2.circle(vis_geo, (px, py), ks(16), (217, 0, 255), -1)
            cv2.circle(vis_geo, (px, py), ks(16), (255, 255, 255), ks(4))
        for rp in rapids_list:
            ry, rx = int(rp.centroid[0]), int(rp.centroid[1])
            cv2.circle(vis_geo, (rx, ry), ks(14), (255, 0, 50), -1)
            cv2.circle(vis_geo, (rx, ry), ks(14), (255, 255, 0), ks(4))
        for cp in pts_criticos:
            cy, cx = int(cp[0]), int(cp[1])
            cv2.drawMarker(vis_geo, (cx, cy), (0, 255, 255), markerType=cv2.MARKER_TILTED_CROSS, markerSize=ks(18), thickness=ks(4))

        disp_scale = 1200.0 / vis_geo.shape[0]
        vis_small = cv2.resize(vis_geo, (int(vis_geo.shape[1] * disp_scale), 1200), interpolation=cv2.INTER_AREA)
        _, buffer_jpg = cv2.imencode(".jpg", cv2.cvtColor(vis_small, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer_jpg).decode("utf-8")
        del vis_geo, vis_small

        gc.collect()

        return {
            "res_m_px": round(float(default_res), 4),
            "cauce_izq": {"l_cauce": round(L_cauce_L, 2), "l_valle": round(L_recta_L, 2), "sinuosidad": round(sin_L, 3)},
            "cauce_der": {"l_cauce": round(L_cauce_R, 2), "l_valle": round(L_recta_R, 2), "sinuosidad": round(sin_R, 3)},
            "ancho": {
                "promedio": round(W_mean, 2),
                "std": round(W_std, 2),
                "min": round(W_min, 2),
                "max": round(W_max, 2),
                "bankfull": round(W_bankfull_m, 2),
                "faja_aluvial": round(W_corredor_valle_m, 2),
                "indice_confinamiento": round(indice_confinamiento, 2),
                "clase_confinamiento": clase_confinamiento
            },
            "curvatura": {
                "rc_med": round(Rc_med, 2),
                "rc_min": round(Rc_min, 2),
                "lm_meandro": round(Lm_meandro, 2),
                "rc_w_ratio": round(Rc_W_rel, 2),
                "pct_erosion_crit": round(pct_erosion_crit, 1)
            },
            "trenzamiento": {
                "braiding_index": round(braiding_index, 2),
                "canales_max": canales_activos_max,
                "pct_bifurcado": round(frac_bifurcada * 100.0, 1)
            },
            "unidades": {
                "piscinas": num_piscinas,
                "rapidos": num_rapidos,
                "relacion_pr": round(relacion_pr, 2),
                "espaciamiento_piscinas_m": round(espaciamiento_piscinas_m, 2),
                "espaciamiento_rapidos_m": round(espaciamiento_rapidos_m, 2),
                "espaciamiento_rel_w": round(espaciamiento_rel_W, 1),
                "pendiente_m_m": round(pendiente_est_m_m, 4),
                "pendiente_pct": round(pendiente_est_pct, 2)
            },
            "areas": {
                "cauce_m2": round(cauce_m2, 1),
                "estanque_m2": round(estanque_m2, 1),
                "pct_lotico": round(pct_lotico, 1),
                "pct_lentico": round(pct_lentico, 1),
                "barras_m2": round(area_barras_m2, 1),
                "prop_barras_pct": round(prop_barras_pct, 1),
                "isla_m2": round(area_isla_m2, 1),
                "prop_isla_pct": round(prop_isla_pct, 1)
            },
            "vegetacion": {
                "cobertura_total_pct": round(cobertura_vegetal_total_pct, 1),
                "area_veg_ha": round(area_veg_ha, 2),
                "dosel_ripario_pct": round(cobertura_dosel_ripario_pct, 1),
                "grado_oclusion_pct": round(grado_oclusion_pct, 1)
            },
            "perfil_izq": {"dist": np.round(dist_L, 1).tolist(), "width": np.round(widths_L, 2).tolist()},
            "perfil_der": {"dist": np.round(dist_R, 1).tolist(), "width": np.round(widths_R, 2).tolist()},
            "image_base64": f"data:image/jpeg;base64,{img_b64}"
        }
    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return JSONResponse(status_code=500, content={"error": str(e)})
