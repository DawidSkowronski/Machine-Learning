import mnist_loader
import math
import numpy as np
import matplotlib.pyplot as plt

def wygeneruj_wagi(wymiar_wejscie, wymiar_wyjscie):
    # wyjściem jest liczba neuronów w następnej warstwie
    wektor_wag = np.random.normal(0,1/math.sqrt(wymiar_wejscie),(1+wymiar_wejscie)*wymiar_wyjscie)
    return np.reshape(wektor_wag, (wymiar_wyjscie, 1+wymiar_wejscie )  )

# Definiujemy funkcje aktywacji fi (sigmoid)
# fi' = fi*(1-fi) 
def sigmoid(x): return (1+np.exp(-x))**(-1) 

def deriv_sigmoid(x): return sigmoid(x)*(1-sigmoid(x))

# Definiujemy funkcję aktywacji ReLU
def relu(x): return np.maximum(0,x) # maximum zwraca macierz

def deriv_relu(x): return np.where(x>0, 1,0)


class Warstwa:
    def __init__(self, liczba_wejscie, liczba_neuronow, funkcja_aktywacji = "sigmoid"):
        """liczba neuronów - w warstwie (czyli liczba danych na wyjściu)"""
        self.liczba_neuronow = liczba_neuronow

        self.liczba_wejscie = liczba_wejscie
        
        # Ustalamy stałą uczenia dla całej warstwy
        self.stala_uczenia = 0.03

        # Inicjalizacja wag
        self.wagi = wygeneruj_wagi(self.liczba_wejscie, self.liczba_neuronow)
        
        # Funkcja aktywacji, która będzie wykorzystana w danej warstwie
        self.funkcja_aktywacji = funkcja_aktywacji

        # Rzeczy do zapamiętania,
        #  Może wystarczy zapamiętać tylko ostatnie wartosci
        self.lista_dane_wejscie = list()
        self.lista_net = list()
    

    def fun_aktywacji(self, x): 
        if self.funkcja_aktywacji == "sigmoid": 
            return sigmoid(x)
        elif self.funkcja_aktywacji == "relu": 
            return relu(x)

    def pochodna_fun_aktywacji(self, x):
        if self.funkcja_aktywacji == "sigmoid": 
            return deriv_sigmoid(x)
        elif self.funkcja_aktywacji == "relu": 
            return deriv_relu(x)

    def forward_prop(self, dane_wejscie ):
        """Definiujemy propagację w przód.
        Podajemy dane na wejście, dodajemy na bias = 1 na początek wektora (otrzymujemy w ten sposób X).
        Zadajemy najpierw ile chcemy mieć neuronów na wyjściu
        Nastepnie liczymy net = W*X <- wyjście netto.
        Póżniej nakładamy funkcję aktywacji fi na net, otrzymując fi(net) = a <- wyjście z warstwy.
        Musimy też zapisywać stany x, net, a dla każdej warstwy (np. listy)"""
        
        rozmiar_batcha = dane_wejscie.shape[1]
        biasy = np.ones((1, rozmiar_batcha))

        self.X = np.vstack([biasy, dane_wejscie]) # Dodajemy biasy

        self.net = self.wagi @ self.X

        dane_wyjscie = self.fun_aktywacji(self.net)

        self.lista_net.append(self.net)
        self.lista_dane_wejscie.append(self.X)

        self.wyjscia_forward_prop = dane_wyjscie

        return dane_wyjscie
    
    def back_prop(self, dL_a):
        """Definiujemy propagację wstecz.
        W ostatniej warstwie liczymy funkcję straty, następnie pochodną dL/da."""
        
        # Funkcja straty to L = 1/2(a[L]-y)^2, czyli pochodna z L to a[L]-y

        # mamy pochodną dL/da * fi(net)
        delta = dL_a * self.pochodna_fun_aktywacji( self.net )
        dL_dW = delta @ self.X.T

        # Liczymy dL/dX i usuwamy pierwszy element, otrzymując dL/da niższej warstwy
        dL_dX = self.wagi.T @ delta
        
        # Dane do przekazania warstwę niżej
        dL_a = dL_dX[1:]

        # Aktualizujemy wagi
        self.wagi = self.wagi - self.stala_uczenia * dL_dW

        return dL_a
    

class SiecNeuronowa:
    def __init__(self, wymiary = [784, 128, 64, 10], funkcje_aktywacji=None):

        # Definiujemy domyślne funkcje aktywacji dla warstw (np. sigmoid)
        if funkcje_aktywacji == None:
            funkcje_aktywacji = ["sigmoid"]*(len(wymiary)-1)

        # definiujemy listę obiektów warstwy
        self.warstwy = list()
        # Może każdej warstwy definiujemy funkcję aktywacji? (W ten sposób można wykorzystać różne funkcje)

        for i in range(len(wymiary)-1):
            warstwa = Warstwa(wymiary[i], wymiary[i+1], funkcje_aktywacji[i]) # Liczba na wejście i na wyjście
            self.warstwy.append(warstwa)
            print(f"Warstwa {i}: {wymiary[i]} -> {wymiary[i+1]}, funkcja {funkcje_aktywacji[i]}")

    def forward_propagation(self, X):
        wejscie = X
        
        for warstwa in self.warstwy:
            wejscie = warstwa.forward_prop(wejscie)

        return wejscie
            

    def backward_propagation(self, y):

        ostatnie_wyjscie = self.warstwy[-1].wyjscia_forward_prop

        dL_da = ostatnie_wyjscie - y

        for i in range(len(self.warstwy) - 1, -1, -1):
            dL_da = self.warstwy[i].back_prop(dL_da)

        return dL_da
    
    def krok_uczenia(self, X, y):
        """Definiujemy kroki dla jednej epoki"""

        # Forward prop
        y_estymowany = self.forward_propagation(X)

        # Strata
        strata = np.mean((y_estymowany - y)**2)

        # Back prop
        self.backward_propagation(y)

        return strata # zwracamy wartośc funkcji straty

    def fit(self, dane_w_krotkach, epoki = 10, rozmiar_batcha = 32):
        """Przeprowadzamy fit() dla wielu epok"""

        straty = list()

        for epoka in range(epoki):

            strata_w_epoce = 0
            liczba_batchy = 0

            # W każdej epoce "mieszamy" dane
            np.random.shuffle(dane_w_krotkach)

            # dzielimy zbiór danych na porcje
            for i in range(0, len(dane_w_krotkach), rozmiar_batcha): 
                batch = dane_w_krotkach[i:i + rozmiar_batcha]

                # Wyciągamy z krotki osobno X i y
                X_batch = np.hstack([x for x, y in batch])
                y_batch = np.hstack([y for x, y in batch])
                

                # Strata (oraz wykonanie forward prop i back prop)
                strata = self.krok_uczenia(X_batch, y_batch)
                strata_w_epoce += strata
                liczba_batchy +=1
            
            srednia_strata = strata_w_epoce/liczba_batchy
            straty.append(srednia_strata)

            
        return straty

    def predict(self, X):
        return self.forward_propagation(X)

    def dokladnosc(self, dane):
        """ Oblicza dokładność klasyfikacji"""
        poprawne = 0
        liczba_danych = len(dane)

        for x, y in dane:

            # Z y daną cyfrę
            y_prawdziwe = np.argmax(y)

            przewidywane = self.predict(x)
            # Wybieramy z wektora argument o najwyższej wartości
            y_przewidywane = np.argmax(przewidywane)

            # Sprawdzamy czy sieć dopasowałą poprawnie etykietę
            if y_prawdziwe == y_przewidywane:
                poprawne +=1
        # Zwracamy dokładność
        return poprawne/liczba_danych
            
def trenuj_siec():
    dane_treningowe, dane_walidacyjne, dane_testowe = mnist_loader.load_data_wrapper()
    # Pomijamy dane walidacyjne
    dane_treningowe = list(dane_treningowe)
    dane_testowe = list(dane_testowe)

    # Tworzymy sieć
    fun_akt_sig = ["sigmoid","sigmoid","sigmoid","sigmoid"]
    fun_akt_relu_sig = ["relu","relu","sigmoid"]
    fun_akt_relu = ["relu","relu","relu"]
    siec = SiecNeuronowa([784,128,64,10], funkcje_aktywacji=fun_akt_sig)

    print("Trening")
    historia = siec.fit(dane_treningowe, epoki= 10, rozmiar_batcha= 100)

    dokladnosc_trening = siec.dokladnosc(dane_treningowe)
    dokladnosc_test = siec.dokladnosc(dane_testowe)

    print(f"Dokładność na zbiorze treningowym: {100*dokladnosc_trening:.4f} %")
    print(f"Dokładność na zbiorze testowym: {100*dokladnosc_test:.4f} %")

    return siec, historia

trenuj_siec()