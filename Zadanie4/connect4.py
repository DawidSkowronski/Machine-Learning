import pygame
import sys
import math
import random

# --- KONFIGURACJA PYGAME ---
pygame.init()
pygame.font.init()

# Wymiary okna
WIDTH, HEIGHT = 700, 700

# Kolory
BLUE = (0, 0, 200)       # Kolor planszy
BLACK = (0, 0, 0)        # Tło (puste dziury)
RED = (200, 0, 0)        # Gracz (Człowiek)
YELLOW = (200, 200, 0)   # Bot (AI)
WHITE = (255, 255, 255)  # Tekst
BTN_COLOR = (50, 50, 50)
BTN_HOVER = (80, 80, 80)

FONT_BIG = pygame.font.SysFont('arial', 50, bold=True)
FONT_MED = pygame.font.SysFont('arial', 30)
FONT_SMALL = pygame.font.SysFont('arial', 20)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Connect 4 AI - Edytowalne')

# --- ZMIENNE GRY ---
GAME_CONFIG = {
    'rows': 6,       # Wiersze
    'cols': 7,       # Kolumny
    'win_len': 4,    # Ile w rzędzie
    'starter': 'ja', # Kto zaczyna
    'ai_depth': 4    # Głębokość myślenia
}

# Reprezentacja planszy: macierz 2D [wiersz][kolumna]
plansza = []

# --- LOGIKA GRY ---

def tworz_plansze(rows, cols):
    return [[0 for _ in range(cols)] for _ in range(rows)]

def jest_prawidlowa_kolumna(board, col):
    return board[0][col] == 0

def znajdz_pierwszy_wolny_wiersz(board, col, rows):
    for r in range(rows - 1, -1, -1):
        if board[r][col] == 0:
            return r
    return None

def wrzuc_zeton(board, row, col, piece):
    board[row][col] = piece

def sprawdz_wygrana(board, piece, rows, cols, win_len):
    # Poziomo
    for c in range(cols - win_len + 1):
        for r in range(rows):
            if all(board[r][c+i] == piece for i in range(win_len)):
                return True

    # Pionowo
    for c in range(cols):
        for r in range(rows - win_len + 1):
            if all(board[r+i][c] == piece for i in range(win_len)):
                return True

    # Ukos /
    for c in range(cols - win_len + 1):
        for r in range(rows - win_len + 1):
            if all(board[r+i][c+i] == piece for i in range(win_len)):
                return True

    # Ukos \
    for c in range(cols - win_len + 1):
        for r in range(win_len - 1, rows):
            if all(board[r-i][c+i] == piece for i in range(win_len)):
                return True
    return False

def czy_remis(board):
    return all(x != 0 for x in board[0])

# --- AI (MINIMAX + HEURYSTYKA) ---

def ocen_okno(window, piece, empty, opp_piece, win_len):
    score = 0
    if window.count(piece) == win_len:
        score += 100
    elif window.count(piece) == win_len - 1 and window.count(empty) == 1:
        score += 5
    elif window.count(piece) == win_len - 2 and window.count(empty) == 2:
        score += 2

    if window.count(opp_piece) == win_len - 1 and window.count(empty) == 1:
        score -= 4 

    return score

def heurystyka(board, piece, rows, cols, win_len):
    score = 0
    opp_piece = 1 if piece == 2 else 2
    
    # Preferuj środek
    center_array = [i for i in list(col[cols//2] for col in board)]
    center_count = center_array.count(piece)
    score += center_count * 3

    # Poziomo
    for r in range(rows):
        row_array = board[r]
        for c in range(cols - win_len + 1):
            window = row_array[c:c+win_len]
            score += ocen_okno(window, piece, 0, opp_piece, win_len)

    # Pionowo
    for c in range(cols):
        col_array = [board[r][c] for r in range(rows)]
        for r in range(rows - win_len + 1):
            window = col_array[r:r+win_len]
            score += ocen_okno(window, piece, 0, opp_piece, win_len)

    # Ukosy
    for r in range(rows - win_len + 1):
        for c in range(cols - win_len + 1):
            window = [board[r+i][c+i] for i in range(win_len)]
            score += ocen_okno(window, piece, 0, opp_piece, win_len)

    for r in range(rows - win_len + 1):
        for c in range(cols - win_len + 1):
            window = [board[r+3-i][c+i] for i in range(win_len)] 
            score += ocen_okno(window, piece, 0, opp_piece, win_len)
    
    return score

def minimax(board, depth, alpha, beta, maximizingPlayer, rows, cols, win_len):
    if sprawdz_wygrana(board, 2, rows, cols, win_len): return 100000000000
    if sprawdz_wygrana(board, 1, rows, cols, win_len): return -100000000000
    if czy_remis(board): return 0
    if depth == 0:
        return heurystyka(board, 2, rows, cols, win_len)

    valid_locations = [c for c in range(cols) if jest_prawidlowa_kolumna(board, c)]
    center = cols // 2
    valid_locations.sort(key=lambda x: abs(x - center))

    if maximizingPlayer:
        value = -math.inf
        for col in valid_locations:
            row = znajdz_pierwszy_wolny_wiersz(board, col, rows)
            b_copy = [x[:] for x in board]
            wrzuc_zeton(b_copy, row, col, 2)
            new_score = minimax(b_copy, depth-1, alpha, beta, False, rows, cols, win_len)
            value = max(value, new_score)
            alpha = max(alpha, value)
            if alpha >= beta: break
        return value
    else:
        value = math.inf
        for col in valid_locations:
            row = znajdz_pierwszy_wolny_wiersz(board, col, rows)
            b_copy = [x[:] for x in board]
            wrzuc_zeton(b_copy, row, col, 1)
            new_score = minimax(b_copy, depth-1, alpha, beta, True, rows, cols, win_len)
            value = min(value, new_score)
            beta = min(beta, value)
            if alpha >= beta: break
        return value

def ruch_bota(rows, cols, win_len):
    best_score = -math.inf
    best_col = random.randint(0, cols-1)
    depth = GAME_CONFIG['ai_depth']
    
    valid_locations = [c for c in range(cols) if jest_prawidlowa_kolumna(plansza, c)]
    
    # Pierwszy ruch na środku (szybkość)
    if len(valid_locations) == cols and all(plansza[r][c] == 0 for r in range(rows) for c in range(cols)):
        return cols // 2

    for col in valid_locations:
        row = znajdz_pierwszy_wolny_wiersz(plansza, col, rows)
        b_copy = [x[:] for x in plansza]
        wrzuc_zeton(b_copy, row, col, 2)
        score = minimax(b_copy, depth, -math.inf, math.inf, False, rows, cols, win_len)
        
        if score > best_score:
            best_score = score
            best_col = col
            
    return best_col

# --- INTERFEJS GRAFICZNY ---

def draw_text_centered(text, font, color, y_off, bg=None):
    surf = font.render(text, True, color, bg)
    rect = surf.get_rect(center=(WIDTH//2, HEIGHT//2 + y_off))
    screen.blit(surf, rect)
    return rect

def draw_button(text, y, w=260):
    mouse_pos = pygame.mouse.get_pos()
    rect = pygame.Rect(WIDTH//2 - w//2, y, w, 50)
    col = BTN_HOVER if rect.collidepoint(mouse_pos) else BTN_COLOR
    pygame.draw.rect(screen, col, rect, border_radius=10)
    surf = FONT_MED.render(text, True, WHITE)
    text_rect = surf.get_rect(center=rect.center)
    screen.blit(surf, text_rect)
    return rect

def rysuj_plansze_gry(rows, cols):
    SQUARESIZE = min(WIDTH // cols, (HEIGHT - 100) // rows)
    offset_y = 100 
    screen.fill(BLACK)
    
    for c in range(cols):
        for r in range(rows):
            rect_x = c * SQUARESIZE + (WIDTH - cols*SQUARESIZE)//2
            rect_y = r * SQUARESIZE + offset_y
            
            pygame.draw.rect(screen, BLUE, (rect_x, rect_y, SQUARESIZE, SQUARESIZE))
            
            color = BLACK
            if plansza[r][c] == 1: color = RED
            elif plansza[r][c] == 2: color = YELLOW
            
            radius = int(SQUARESIZE / 2 - 5)
            pygame.draw.circle(screen, color, (rect_x + SQUARESIZE//2, rect_y + SQUARESIZE//2), radius)

    return SQUARESIZE, offset_y

# --- MAIN ---

def main():
    global plansza  # <--- TO NAPRAWIA PROBLEM INDEXERROR
    
    stan = "MENU"
    tura = 0 
    game_over = False
    wynik_msg = ""
    clock = pygame.time.Clock()

    while True:
        events = pygame.event.get()
        for e in events:
            if e.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if stan == "MENU":
            screen.fill(BLUE)
            draw_text_centered("CONNECT 4 - AI", FONT_BIG, WHITE, -250)
            
            # Przyciski
            if draw_button(f"Wiersze: {GAME_CONFIG['rows']}", 100).collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
                pygame.time.delay(150)
                GAME_CONFIG['rows'] = 4 if GAME_CONFIG['rows'] >= 10 else GAME_CONFIG['rows'] + 1

            if draw_button(f"Kolumny: {GAME_CONFIG['cols']}", 170).collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
                pygame.time.delay(150)
                GAME_CONFIG['cols'] = 4 if GAME_CONFIG['cols'] >= 10 else GAME_CONFIG['cols'] + 1

            if draw_button(f"Wygrywa: {GAME_CONFIG['win_len']}", 240).collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
                pygame.time.delay(150)
                l = GAME_CONFIG['win_len'] + 1
                if l > min(GAME_CONFIG['rows'], GAME_CONFIG['cols']): l = 3
                GAME_CONFIG['win_len'] = l

            if draw_button(f"Zaczyna: {GAME_CONFIG['starter'].upper()}", 310).collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
                pygame.time.delay(150)
                GAME_CONFIG['starter'] = 'bot' if GAME_CONFIG['starter'] == 'ja' else 'ja'

            # Start
            rect_start = pygame.Rect(WIDTH//2 - 100, 450, 200, 80)
            col = (0, 200, 0) if rect_start.collidepoint(pygame.mouse.get_pos()) else (0, 150, 0)
            pygame.draw.rect(screen, col, rect_start, border_radius=15)
            surf = FONT_BIG.render("GRAJ!", True, WHITE)
            screen.blit(surf, surf.get_rect(center=rect_start.center))

            if rect_start.collidepoint(pygame.mouse.get_pos()) and pygame.mouse.get_pressed()[0]:
                # Tworzymy planszę (używając globalnej zmiennej)
                plansza = tworz_plansze(GAME_CONFIG['rows'], GAME_CONFIG['cols'])
                tura = 0 if GAME_CONFIG['starter'] == 'ja' else 1
                game_over = False
                stan = "GRA"

        elif stan == "GRA":
            rows = GAME_CONFIG['rows']
            cols = GAME_CONFIG['cols']
            win_len = GAME_CONFIG['win_len']
            
            sq_size, off_y = rysuj_plansze_gry(rows, cols)
            margin_x = (WIDTH - cols*sq_size)//2

            # Kursor gracza
            if not game_over and tura == 0:
                mx = pygame.mouse.get_pos()[0]
                if mx < margin_x: mx = margin_x
                if mx > margin_x + cols*sq_size: mx = margin_x + cols*sq_size
                pygame.draw.circle(screen, RED, (mx, off_y // 2), sq_size//2 - 5)

            # Kliknięcie gracza
            for e in events:
                if e.type == pygame.MOUSEBUTTONDOWN and not game_over and tura == 0:
                    mx = e.pos[0]
                    col = (mx - margin_x) // sq_size
                    
                    if 0 <= col < cols and jest_prawidlowa_kolumna(plansza, col):
                        row = znajdz_pierwszy_wolny_wiersz(plansza, col, rows)
                        wrzuc_zeton(plansza, row, col, 1)
                        
                        if sprawdz_wygrana(plansza, 1, rows, cols, win_len):
                            wynik_msg = "WYGRAŁEŚ!"
                            game_over = True
                            stan = "KONIEC"
                        elif czy_remis(plansza):
                            wynik_msg = "REMIS!"
                            game_over = True
                            stan = "KONIEC"
                        
                        tura = 1

            # Ruch Bota
            if tura == 1 and not game_over and stan == "GRA":
                pygame.display.update() # Odśwież żeby pokazać ruch gracza
                # pygame.time.delay(100)
                
                col = ruch_bota(rows, cols, win_len)
                
                if col is not None and jest_prawidlowa_kolumna(plansza, col):
                    row = znajdz_pierwszy_wolny_wiersz(plansza, col, rows)
                    wrzuc_zeton(plansza, row, col, 2)
                    
                    if sprawdz_wygrana(plansza, 2, rows, cols, win_len):
                        wynik_msg = "BOT WYGRAŁ!"
                        game_over = True
                        stan = "KONIEC"
                    elif czy_remis(plansza):
                        wynik_msg = "REMIS!"
                        game_over = True
                        stan = "KONIEC"
                    
                    tura = 0

        elif stan == "KONIEC":
            s = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            s.fill((0, 0, 0, 180))
            screen.blit(s, (0, 0))
            
            color = RED if "WYGRAŁEŚ" in wynik_msg else YELLOW
            if "REMIS" in wynik_msg: color = WHITE
            
            draw_text_centered(wynik_msg, FONT_BIG, color, -50)
            draw_text_centered("SPACJA - MENU", FONT_MED, WHITE, 50)

            for e in events:
                if e.type == pygame.KEYDOWN:
                    if e.key == pygame.K_SPACE:
                        stan = "MENU"

        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    main()