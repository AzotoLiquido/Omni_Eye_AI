"""
Test Rate Limiting - Omni Eye AI
Verifica che i limiti funzionino correttamente
"""

import requests
import time
from colorama import init, Fore, Style

init(autoreset=True)

BASE_URL = "http://localhost:5000"

def test_endpoint(endpoint, limit_per_min, test_name):
    """Testa un endpoint specifico"""
    print(f"\n{'='*60}")
    print(f"{Fore.CYAN}TEST: {test_name}")
    print(f"Endpoint: {endpoint}")
    print(f"Limite configurato: {limit_per_min} richieste/minuto")
    print(f"{'='*60}\n")
    
    # Calcola quante richieste inviare (oltre il limite)
    requests_to_send = int(limit_per_min * 1.2)  # 20% oltre il limite
    
    print(f"Invio {requests_to_send} richieste in rapida successione...")
    
    blocked = False
    blocked_at = 0
    
    for i in range(requests_to_send):
        try:
            if endpoint == "/api/chat/stream":
                r = requests.post(
                    BASE_URL + endpoint,
                    json={"message": f"test {i}", "conversation_id": None},
                    timeout=5
                )
            elif endpoint == "/api/status":
                r = requests.get(BASE_URL + endpoint, timeout=5)
            else:
                r = requests.get(BASE_URL + endpoint, timeout=5)
            
            if r.status_code == 429:
                blocked = True
                blocked_at = i + 1
                print(f"\n{Fore.GREEN}✅ RATE LIMIT FUNZIONA!")
                print(f"{Fore.YELLOW}Bloccato alla richiesta #{blocked_at}")
                print(f"\nRisposta server:")
                print(f"{Fore.CYAN}{r.json()}")
                break
            elif i % 10 == 0:
                print(f"Richiesta #{i+1}: {Fore.GREEN}OK (200)")
                
        except requests.exceptions.Timeout:
            print(f"Richiesta #{i+1}: {Fore.YELLOW}Timeout (normale per stream)")
        except Exception as e:
            print(f"Richiesta #{i+1}: {Fore.RED}Errore - {e}")
        
        time.sleep(0.05)  # 50ms tra richieste = 1200 req/min
    
    if not blocked:
        print(f"\n{Fore.RED}⚠️  RATE LIMIT NON ATTIVO")
        print(f"Tutte le {requests_to_send} richieste sono passate!")
        print(f"Controlla che RATE_LIMIT_ENABLED=True in config")
    
    return blocked

def main():
    print(f"\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}🛡️  TEST RATE LIMITING - OMNI EYE AI")
    print(f"{Fore.MAGENTA}{'='*60}\n")
    
    # Verifica che il server sia attivo
    try:
        r = requests.get(BASE_URL + "/api/status", timeout=3)
        print(f"{Fore.GREEN}✅ Server online\n")
    except Exception:
        print(f"{Fore.RED}❌ Server offline!")
        print(f"Avvia il server con: python start.py")
        return
    
    # Test vari endpoint
    tests = [
        ("/api/status", 200, "Endpoint Status (limite default)"),
        # ("/api/chat/stream", 60, "Chat Streaming (limite specifico)"),
        # Decommentare per testare altri endpoint
    ]
    
    results = []
    for endpoint, limit, name in tests:
        result = test_endpoint(endpoint, limit, name)
        results.append((name, result))
        time.sleep(2)  # Pausa tra test
    
    # Riepilogo
    print(f"\n\n{Fore.MAGENTA}{'='*60}")
    print(f"{Fore.MAGENTA}RIEPILOGO TEST")
    print(f"{Fore.MAGENTA}{'='*60}\n")
    
    for name, passed in results:
        status = f"{Fore.GREEN}✅ PASS" if passed else f"{Fore.RED}❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\n{Fore.CYAN}NOTA: Se vuoi disabilitare il rate limiting:")
    print(f"{Fore.YELLOW}  1. Crea file .env")
    print(f"{Fore.YELLOW}  2. Aggiungi: RATE_LIMIT_ENABLED=False")
    print(f"{Fore.YELLOW}  3. Riavvia il server\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}Test interrotto dall'utente")
