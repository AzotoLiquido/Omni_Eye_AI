"""
Omni Eye AI - Script di Avvio Rapido
"""

import sys
import os
import subprocess
import time

# Aggiungi la directory corrente al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_banner():
    """Stampa il banner di avvio"""
    print("\n" + "=" * 70)
    print("🤖  OMNI EYE AI - Assistente AI Locale")
    print("=" * 70)
    print("    Intelligenza Artificiale completamente privata e gratuita")
    print("    Tutto resta sul tuo PC - Nessun limite di utilizzo")
    print("=" * 70 + "\n")

def check_python_version():
    """Verifica la versione di Python"""
    print("🔍 Verifica Python...")
    
    if sys.version_info < (3, 8):
        print("   ❌ Python 3.8+ richiesto!")
        print(f"   Versione attuale: {sys.version}")
        return False
    
    print(f"   ✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_dependencies():
    """Verifica le dipendenze Python"""
    print("\n🔍 Verifica dipendenze...")
    
    required = ['flask', 'ollama', 'PyPDF2', 'docx', 'flask_cors']
    missing = []
    
    for package in required:
        try:
            if package == 'docx':
                __import__('docx')
            else:
                __import__(package)
            print(f"   ✅ {package}")
        except ImportError:
            print(f"   ❌ {package} mancante")
            missing.append(package)
    
    if missing:
        print("\n⚠️  Dipendenze mancanti trovate!")
        response = input("Vuoi installarle automaticamente? (s/n): ")
        
        if response.lower() == 's':
            print("\n📦 Installazione dipendenze...")
            try:
                subprocess.check_call([
                    sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
                ])
                print("✅ Dipendenze installate!")
                return True
            except subprocess.CalledProcessError:
                print("❌ Errore nell'installazione")
                return False
        else:
            print("\n💡 Installa manualmente con: pip install -r requirements.txt")
            return False
    
    return True

def check_ollama():
    """Verifica se Ollama è installato e disponibile"""
    print("\n🔍 Verifica Ollama...")
    
    # Prova prima il comando diretto, poi il percorso standard Windows
    ollama_commands = [
        'ollama',
        os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Programs', 'Ollama', 'ollama.exe')
    ]
    
    for ollama_cmd in ollama_commands:
        try:
            result = subprocess.run(
                [ollama_cmd, 'list'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                print("   ✅ Ollama disponibile")
                
                # Mostra modelli installati
                models = result.stdout.strip().split('\n')[1:]  # Salta header
                if models and models[0]:
                    print("\n   📦 Modelli installati:")
                    for model in models[:5]:  # Max 5
                        if model.strip():
                            model_name = model.split()[0]
                            print(f"      - {model_name}")
                else:
                    print("\n   ⚠️  Nessun modello installato!")
                    suggest_model_download()
                    
                return True
                
        except (FileNotFoundError, OSError):
            continue
        except subprocess.TimeoutExpired:
            print("   ❌ Timeout - Ollama non risponde")
            return False
        except Exception as e:
            continue
    
    # Se arriviamo qui, ollama non è stato trovato
    print("   ❌ Ollama non trovato!")
    print("\n   💡 Installa Ollama:")
    print("      Windows: winget install Ollama.Ollama")
    print("      Oppure: https://ollama.ai/download")
    return False

def suggest_model_download():
    """Suggerisce di scaricare un modello"""
    print("\n💡 Scarica un modello AI:")
    print("   Modelli consigliati per iniziare:")
    print("   - llama3.2 (3B) - Veloce, buono per PC normali")
    print("   - mistral (7B) - Ottimo bilanciamento")
    print("   - llama3.1 (8B) - Molto intelligente")
    
    response = input("\nVuoi scaricare llama3.2 ora? (s/n): ")
    
    if response.lower() == 's':
        print("\n⏳ Download del modello... (può richiedere alcuni minuti)")
        try:
            subprocess.run(['ollama', 'pull', 'llama3.2'], check=True)
            print("✅ Modello scaricato!")
            return True
        except subprocess.CalledProcessError:
            print("❌ Errore nel download")
            return False
    else:
        print("\n💡 Scarica manualmente: ollama pull llama3.2")
        return False

def start_server():
    """Avvia il server Flask"""
    print("\n🚀 Avvio server...")
    print("=" * 70)
    
    # Importa e avvia l'app
    try:
        from app.main import app
        import config
        
        print(f"\n🌐 Server in esecuzione su: http://localhost:{config.SERVER_CONFIG['port']}")
        print("=" * 70)
        print("\n💡 Premi Ctrl+C per fermare il server\n")
        
        app.run(
            host=config.SERVER_CONFIG['host'],
            port=config.SERVER_CONFIG['port'],
            debug=config.SERVER_CONFIG['debug']
        )
    except KeyboardInterrupt:
        print("\n\n👋 Arresto del server...")
        print("   Grazie per aver usato Omni Eye AI!")
    except Exception as e:
        print(f"\n❌ Errore avvio server: {e}")
        print("\n💡 Verifica che tutte le dipendenze siano installate")
        return False

def main():
    """Funzione principale"""
    print_banner()
    
    # Verifica Sistema
    if not check_python_version():
        input("\nPremi Enter per uscire...")
        sys.exit(1)
    
    if not check_dependencies():
        input("\nPremi Enter per uscire...")
        sys.exit(1)
    
    ollama_ok = check_ollama()
    
    if not ollama_ok:
        print("\n⚠️  Ollama è necessario per far funzionare Omni Eye AI")
        response = input("Vuoi continuare comunque? (s/n): ")
        if response.lower() != 's':
            sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✅ Sistema pronto!")
    print("=" * 70)
    
    input("\nPremi Enter per avviare il server...")
    
    # Avvia server
    start_server()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Arrivederci!")
    except Exception as e:
        print(f"\n❌ Errore critico: {e}")
        input("\nPremi Enter per uscire...")
