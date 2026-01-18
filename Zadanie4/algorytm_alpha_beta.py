#Musimy mieć funkcje:
# koniec(stan) == bool
# wartosc(stan) = -1 0 1
# gracz(stan) = MAX MIN
# akcje(stan) = lista możliwych ruchów
# wynik(stan,akcja) = zwraca wynik akcji

import math

# Załóżmy że my jesteśmy X i jesteśmy graczem MAX
KTO_ZACZYNA = "MAX"

plansza = [' ' for i in range(0,9)]
print(plansza)

def drukuj_plansze(plansza):
    print(f"\n {plansza[0]} | {plansza[1]} | {plansza[2]} ")
    print("---+---+---")
    print(f" {plansza[3]} | {plansza[4]} | {plansza[5]} ")
    print("---+---+---")
    print(f" {plansza[6]} | {plansza[7]} | {plansza[8]} \n")


def czy_wygral(stan, gracz):
    poz_wygrywajace = [(0,1,2),(3,4,5),(6,7,8), # wiersze
                       (0,3,6),(1,4,7),(2,5,8), # kolumny
                       (0,4,8),(2,4,6) # diagonale
                       ]
    for a,b,c in poz_wygrywajace:
        if stan[a] == stan[b] == stan[c]==gracz:
            return True
    return False

def koniec(stan):
    # Gdy wygrał X
    if czy_wygral(stan,"X"):
        return True
    # Gdy wygrał O
    if czy_wygral(stan, "O"):
        return True
    # Jeśli nie ma pustych pól tzn. że jest remis
    if ' ' not in stan:
        return True
    # Gra się jeszcze nie skończyła
    return False

def wartosc(stan):
    if czy_wygral(stan,"X"):
        return 1 # wygrał człowiek MAX
    if czy_wygral(stan,"O"):
        return -1 # wygrał bot MIN
    else:
        return 0 # remis

def gracz(stan):
    """Sprawdzamy, który gracz jest na ruchu"""
    liczba_x = stan.count("X")
    liczba_o = stan.count("O")
    suma_ruchow = liczba_o + liczba_x

    # Jeśli liczba ruchów jest parzysta, to ruch ma ten, kto zaczął grę
    if suma_ruchow % 2 == 0:
        return KTO_ZACZYNA
    else:
        return "MAX" if KTO_ZACZYNA == "MIN" else "MIN"

def akcje(stan):
    mozliwe_ruchy = list()
    for i in range(9):
        if stan[i] == ' ':
            mozliwe_ruchy.append(i)
    return mozliwe_ruchy

def wynik(stan,akcja):
    # Tworzymy kopię stanu, aby nie popsuć minmax
    nowy_stan = stan.copy()
    kto_rusza = gracz(stan)

    symbol = "O" if kto_rusza == "MIN" else "X"

    nowy_stan[akcja] = symbol
    return nowy_stan

def minimax(stan):
    # Sprawdzamy, czy koniec gry
    if koniec(stan) == True:
        return wartosc(stan)
    
    if gracz(stan) == "MAX":
        wart = -math.inf
        for akcja in akcje(stan):
            wart = max(wart, minimax(wynik(stan, akcja)))
        return wart
    
    if gracz(stan) == "MIN":
        wart = math.inf
        for akcja in akcje(stan):
            wart = min(wart, minimax(wynik(stan, akcja)))
        return wart
    
def ruch_bota(stan):
    najlepszy_wynik = math.inf # bot jest graczem MIN
    najlepsza_akcja = None

    # Sprawdzamy dostępne ruchy
    for akcja in akcje(stan):
        stan_po_ruchu = wynik(stan, akcja)
        wartosc_ruchu = minimax(stan_po_ruchu)

        # Bot szuka minimum
        if wartosc_ruchu < najlepszy_wynik:
            najlepszy_wynik = wartosc_ruchu
            najlepsza_akcja = akcja
    
    return najlepsza_akcja

# GRA
print("Ruch gracza 'X'")
akutalna_plansza = plansza

while not koniec(akutalna_plansza):
    drukuj_plansze(akutalna_plansza)
    kolej = gracz(akutalna_plansza)

    if kolej == "MAX": # ruch gracza
        try:
            wybor = int(input("Wybierz pole (1-9): ")) - 1
            if akutalna_plansza[wybor] == ' ':
                akutalna_plansza = wynik(akutalna_plansza,wybor)
            else:
                print("To pole jest zajęte")
        except (ValueError, IndexError):
            print("Nieprawidłowy numer pola")

    else: # ruch bota
        print("bot myśli...")
        ruch = ruch_bota(akutalna_plansza)
        akutalna_plansza = wynik(akutalna_plansza,ruch)
    
drukuj_plansze(akutalna_plansza)
wart = wartosc(akutalna_plansza)

if wart == 1: ("Wygrał gracz")
elif wart == -1: print("Wygrał bot")
elif wart == 0: print("Remis")
