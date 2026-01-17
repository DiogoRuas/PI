import cv2
import numpy as np
import matplotlib.pyplot as plt

#G4
def processamento_da_imagem(file_name):
    img_rgb = plt.imread(file_name)
    
    #Preparar Imagem
    escala = 1200 / max(img_rgb.shape[:2])
    img_rgb = cv2.resize(img_rgb, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)

    # Converter para HSV
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # --- Segmentar vermelho ---
    mask_r1 = cv2.inRange(img_hsv, (0, 80, 60), (10, 255, 255))
    mask_r2 = cv2.inRange(img_hsv, (170, 80, 60), (180, 255, 255))
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    # --- Segmentar branco ---
    mask_white = cv2.inRange(img_hsv, (0, 0, 160), (180, 80, 255))

    mask_red = cv2.medianBlur(mask_red, 5)
    mask_white = cv2.medianBlur(mask_white, 5)

    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))

    mask_r_d = cv2.dilate(mask_red, ker, iterations = 1)
    mask_w_d = cv2.dilate(mask_white, ker, iterations = 1)

    inter = cv2.bitwise_and(mask_r_d, mask_w_d)

    num_r, labels_r = cv2.connectedComponents(mask_r_d, connectivity = 8)

    num_w, labels_w = cv2.connectedComponents(mask_w_d)
    
    ids_r = np.unique(labels_r[inter > 0])
    ids_w = np.unique(labels_w[inter > 0])
    
    # Remover fundo (0)
    ids_r = ids_r[ids_r != 0]
    ids_w = ids_w[ids_w != 0]

    # === GERAR MÁSCARAS DE COMPONENTES QUE TOCAM ===
    mask_r_touch = np.zeros_like(labels_r, dtype=np.uint8)
    for rid in ids_r: 
        mask_r_touch |= (labels_r == rid)
    mask_r_touch = (mask_r_touch * 255).astype(np.uint8)
    
    mask_w_touch = np.zeros_like(labels_w, dtype=np.uint8)
    for wid in ids_w: 
        mask_w_touch |= (labels_w == wid)
    mask_w_touch = (mask_w_touch * 255).astype(np.uint8)
    
    # === UNIÃO DAS DUAS REGIÕES (VERMELHO + BRANCO) ===
    mask_tiles_touch = cv2.bitwise_or(mask_r_touch, mask_w_touch)
    
    # Pequena erosão para suavizar
    mask = cv2.erode(mask_tiles_touch, ker, iterations=1)

    
    # === CONTORNOS E POLÍGONO FINAL ===
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError("Nenhum contorno encontrado.")
    
    largest = max(contours, key=cv2.contourArea)
    perimetro = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.01 * perimetro, True)
    
    mask_refinada = np.zeros_like(mask)
    cv2.drawContours(mask_refinada, [approx], -1, 255, thickness=-1)

    # 1. Find contours (each contour is a set of points around a blob)
    contours, hierarchy = cv2.findContours(mask_refinada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Take the largest contour
    contour = max(contours, key=cv2.contourArea)

    # 3. Approximate polygon
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    points = approx.reshape(-1, 2)

    # 4. Compute all line segments
    segments = []
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        length = np.linalg.norm(p1 - p2)
        segments.append((length, p1, p2))

    # 5. Keep 4 longest segments
    segments.sort(key=lambda x: x[0], reverse=True)
    top4 = segments[:4]

    # 6. Function to find intersection between two lines
    def line_intersection(p1, p2, p3, p4):
        """Return intersection point of lines (p1,p2) and (p3,p4) if exists."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
        if abs(denom) < 1e-6:
            return None
        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom
        return np.array([px, py])

    # 7. Compute intersections between all pairs of top4 lines
    intersections = []
    for i in range(len(top4)):
        for j in range(i+1, len(top4)):
            inter = line_intersection(top4[i][1], top4[i][2], top4[j][1], top4[j][2])
            if inter is not None:
                intersections.append(inter)

    # 8. Compute centroid of the polygon
    M = cv2.moments(approx)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    centroid = np.array([cx, cy])

    # 9. Choose 4 closest intersection points to centroid
    if len(intersections) > 4:
        intersections = sorted(intersections, key=lambda p: np.linalg.norm(p - centroid))[:4]

    # 10. Draw results
    overlay = img_rgb.copy()

    for pt in intersections:
        cv2.circle(overlay, tuple(np.int32(pt)), 6, (0,255,0), -1)

    cv2.circle(overlay, (cx, cy), 6, (255,0,0), -1)  # centroid
    for (_, p1, p2) in top4:
        cv2.line(overlay, tuple(p1), tuple(p2), (0,0,255), 2)

    alpha = 0.6
    output = cv2.addWeighted(img_rgb, 1 - alpha, overlay, alpha, 0)

    # === ORDENAR OS 4 CANTOS DETETADOS ===
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]   # top-left
        rect[2] = pts[np.argmax(s)]   # bottom-right

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left

        return rect

    # Converter lista de interseções para matriz
    corners = np.array(intersections, dtype="float32")

    # Ordenar
    rect = order_points(corners)

    # === DEFINIR DESTINO DA TRANSFORMAÇÃO PERSPECTIVA ===
    dst = np.array([
        [100, 100],
        [600, 100],
        [600, 600],
        [100, 600]
    ], dtype="float32")

    # === APLICAR TRANSFORMAÇÃO À IMAGEM FINAL OUTPUT ===
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(output, M, (700, 700))
    board = warped[100:600, 100:600]

    board_hsv = cv2.cvtColor(board, cv2.COLOR_RGB2HSV)

    H = board_hsv[:,:,0]
    S = board_hsv[:,:,1]
    V = board_hsv[:,:,2]

    lower_gold = (9, 10, 10)
    median_hue = np.median(H)
    if median_hue <= 14:
        lower_gold = (6, 0, 0)

    if median_hue > 100:
        lower_gold = (3, 0, 0)
        
    upper_gold = (35, 255, 255) 
    mask_gold = cv2.inRange(board_hsv, lower_gold, upper_gold)

    kerneldlt = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    mask_gold_dlt = cv2.dilate(mask_gold, kerneldlt, iterations = 1)

    mask_gold_clean = cv2.medianBlur(mask_gold_dlt, 5)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8,8))
    mask_gold_clean = cv2.morphologyEx(mask_gold_clean, cv2.MORPH_OPEN, kernel, iterations=1)


    # === CARREGAR TEMPLATES ===
    templates = {}
    for k in range(1, 16):
        t = cv2.imread(f"15GameMasks/mask{k}.png", cv2.IMREAD_GRAYSCALE)
        if t is None:
            print(f"AVISO: Template {k} não encontrado!")
            continue
        templates[k] = t

    # === CRIAR IMAGEM PARA VISUALIZAÇÃO ===
    board_visualization = board.copy()

    # === MATRIZ PARA GUARDAR OS NÚMEROS DETECTADOS ===
    detected_board = np.zeros((4, 4), dtype=int)

    # === PROCESSAR CADA CÉLULA 4x4 ===
    cell_h = board.shape[0] // 4
    cell_w = board.shape[1] // 4

    empty_tile_position = None
    numbers_scores = list()

    for row in range(4):
        for col in range(4):
            # Definir região da célula
            y1 = row * cell_h
            y2 = (row + 1) * cell_h
            x1 = col * cell_w
            x2 = (col + 1) * cell_w
            
            # Extrair máscara da célula
            cell_mask = mask_gold_clean[y1:y2, x1:x2]
            
            # Encontrar contornos na célula
            contours_cell, _ = cv2.findContours(cell_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filtrar contornos por área mínima
            valid_contours = [c for c in contours_cell if cv2.contourArea(c) > 50]
            
            if valid_contours:
                # Há número nesta célula - fazer template matching
                
                # Calcular bounding box que engloba TODOS os contornos
                all_points = []
                for contour in valid_contours:
                    all_points.extend(contour.reshape(-1, 2))
                
                all_points = np.array(all_points)
                
                x_min = np.min(all_points[:, 0])
                y_min = np.min(all_points[:, 1])
                x_max = np.max(all_points[:, 0])
                y_max = np.max(all_points[:, 1])
                
                # Extrair região do número (com pequena margem)
                margin = 5
                x_min_m = max(0, x_min - margin)
                y_min_m = max(0, y_min - margin)
                x_max_m = min(cell_mask.shape[1], x_max + margin)
                y_max_m = min(cell_mask.shape[0], y_max + margin)
                
                number_region = cell_mask[y_min_m:y_max_m, x_min_m:x_max_m]
                
                # === TEMPLATE MATCHING ===
                best_match_score = -1
                best_match_num = 0
                
                for num, template in templates.items():
                    # Redimensionar template para o tamanho da região
                    if number_region.shape[0] > 0 and number_region.shape[1] > 0:
                        template_resized = cv2.resize(template, 
                                                    (number_region.shape[1], number_region.shape[0]),
                                                    interpolation=cv2.INTER_AREA)
                        
                        # Usar TM_CCOEFF_NORMED (valores entre -1 e 1, quanto maior melhor)
                        result = cv2.matchTemplate(number_region, template_resized, cv2.TM_CCOEFF_NORMED)
                        score = result[0, 0]
                        
                        if score > best_match_score:
                            best_match_score = score
                            best_match_num = num
                
                # Guardar na matriz
                detected_board[row, col] = best_match_num
                numbers_scores.append([best_match_num, float(best_match_score), row, col])
                
                # Calcular centroide
                cx = (x_min + x_max) // 2 + x1
                cy = (y_min + y_max) // 2 + y1
                
                # Ajustar coordenadas para o board completo
                x_abs = x1 + x_min
                y_abs = y1 + y_min
                w_abs = x_max - x_min
                h_abs = y_max - y_min
                
                # Desenhar centroide
                cv2.circle(board_visualization, (cx, cy), 5, (255, 0, 255), -1)
                
                # Desenhar retângulo ao redor do número completo
                cv2.rectangle(board_visualization, 
                            (x_abs, y_abs), 
                            (x_abs + w_abs, y_abs + h_abs), 
                            (0, 255, 0), 2)
                
                # Adicionar número detectado
                cv2.putText(board_visualization, f"{best_match_num}", 
                        (cx - 10, cy + 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Adicionar score (para debug)
                cv2.putText(board_visualization, f"{best_match_score:.2f}", 
                        (x_abs, y_abs - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            else:
                # Esta é a célula vazia
                detected_board[row, col] = 0  # 0 representa espaço vazio
                numbers_scores.append([0, -999.0, row, col])  # Adicionar também à lista
                
                # Calcular centro da célula vazia
                cx_empty = x1 + cell_w // 2
                cy_empty = y1 + cell_h // 2
                
                # Desenhar X vermelho grande
                margin = 15
                cv2.line(board_visualization, 
                        (x1 + margin, y1 + margin), 
                        (x2 - margin, y2 - margin), 
                        (0, 0, 255), 4)
                cv2.line(board_visualization, 
                        (x2 - margin, y1 + margin), 
                        (x1 + margin, y2 - margin), 
                        (0, 0, 255), 4)
                
                # Desenhar retângulo da célula vazia
                cv2.rectangle(board_visualization, 
                            (x1 + 5, y1 + 5), 
                            (x2 - 5, y2 - 5), 
                            (0, 0, 255), 3)
                
                # Adicionar texto "0"
                cv2.putText(board_visualization, "0", 
                        (cx_empty - 10, cy_empty + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # === VERIFICAR NÚMEROS REPETIDOS E ENCONTRAR ESPAÇO VAZIO ===
    seen = set()
    duplicates = set()

    # Identificar quais números estão duplicados (excluindo 0)
    for num_scores in numbers_scores:
        if num_scores[0] == 0:  # Ignorar o 0
            continue
        if num_scores[0] in seen:
            duplicates.add(num_scores[0])
        else:
            seen.add(num_scores[0])

    empty_tile_position = None

    if duplicates:
        print(f"Números duplicados encontrados: {duplicates}")
        
        for dup_num in duplicates:
            # Filtrar apenas as ocorrências deste número
            occurrences = [info for info in numbers_scores if info[0] == dup_num]
            
            # Encontrar a ocorrência com menor score
            min_occurrence = min(occurrences, key=lambda x: x[1])
            empty_num, empty_score, empty_row, empty_col = min_occurrence
            
            print(f"Espaço vazio identificado em ({empty_row},{empty_col}) com score {empty_score:.4f}")
            
            # Marcar como espaço vazio
            detected_board[empty_row, empty_col] = 0
            empty_tile_position = (empty_row, empty_col)
    else:
        # Se não houver duplicados, procurar o 0 na lista
        for info in numbers_scores:
            if info[0] == 0:
                empty_tile_position = (info[2], info[3])
                break

    print(f"\nEspaço vazio encontrado na posição: {empty_tile_position}")
    print(f"\nTabuleiro detectado (0 = vazio):")
    print(detected_board)

    # --- Mostrar resultados ---
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 4, 1)
    plt.imshow(img_rgb)
    plt.title("Board Original")
    plt.axis("off")


    plt.subplot(1, 4, 2)
    plt.imshow(board)
    plt.title("Board Original")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(mask_gold_clean, cmap='gray')
    plt.title("Máscara Gold Limpa")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(board_visualization)
    plt.title("Números Detectados via Template Matching")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Flatten the matrix
    flat = [num for row in detected_board for num in row]

    # Check if it contains exactly numbers 0–15
    if set(flat) == set(range(16)):
        return detected_board
    else:
        return None
