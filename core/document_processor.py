"""
Processore Documenti - Estrae testo da vari formati di file
"""

import logging
import os
import time
from typing import Optional, Tuple
import config

logger = logging.getLogger(__name__)

# Import condizionali per gestire dipendenze opzionali
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


class DocumentProcessor:
    """Processa e estrae testo da vari formati di documento"""
    
    def __init__(self):
        """Inizializza il processore di documenti"""
        self.uploads_dir = config.UPLOADS_DIR
        self.max_file_size = config.UPLOAD_CONFIG['max_file_size']
        self.allowed_extensions = config.UPLOAD_CONFIG['allowed_extensions']
    
    def is_allowed_file(self, filename: str) -> bool:
        """
        Verifica se il file è di un tipo supportato
        
        Args:
            filename: Nome del file
            
        Returns:
            True se il file è supportato
        """
        ext = os.path.splitext(filename)[1].lower()
        return ext in self.allowed_extensions
    
    def process_file(self, filepath: str) -> Tuple[str, Optional[str]]:
        """
        Processa un file e estrae il testo
        
        Args:
            filepath: Percorso del file
            
        Returns:
            Tupla (testo_estratto, messaggio_errore)
        """
        if not os.path.exists(filepath):
            return None, "File non trovato"
        
        # Controlla dimensione
        file_size = os.path.getsize(filepath)
        if file_size > self.max_file_size:
            return None, f"File troppo grande (max {self.max_file_size // (1024*1024)}MB)"
        
        # Determina il tipo di file
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()
        
        try:
            if ext == '.txt':
                return self._process_txt(filepath)
            elif ext == '.pdf':
                return self._process_pdf(filepath)
            elif ext == '.docx':
                return self._process_docx(filepath)
            elif ext == '.md':
                return self._process_txt(filepath)  # Markdown come testo
            elif ext in ['.py', '.js', '.html', '.css', '.json']:
                return self._process_code(filepath)
            else:
                return None, f"Tipo di file non supportato: {ext}"
                
        except Exception as e:
            return None, f"Errore durante il processing: {str(e)}"
    
    def _process_txt(self, filepath: str) -> Tuple[str, None]:
        """Processa file di testo"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read(), None
    
    def _process_pdf(self, filepath: str) -> Tuple[Optional[str], Optional[str]]:
        """Processa file PDF"""
        if not PDF_AVAILABLE:
            return None, "PyPDF2 non installato. Installa con: pip install PyPDF2"
        
        try:
            text = []
            with open(filepath, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text.append(page.extract_text())
            
            full_text = '\n'.join(text)
            if not full_text.strip():
                return None, "Nessun testo estratto dal PDF (potrebbe essere solo immagini)"
            
            return full_text, None
            
        except Exception as e:
            return None, f"Errore lettura PDF: {str(e)}"
    
    def _process_docx(self, filepath: str) -> Tuple[Optional[str], Optional[str]]:
        """Processa file DOCX"""
        if not DOCX_AVAILABLE:
            return None, "python-docx non installato. Installa con: pip install python-docx"
        
        try:
            doc = Document(filepath)
            text = []
            
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            
            full_text = '\n'.join(text)
            return full_text, None
            
        except Exception as e:
            return None, f"Errore lettura DOCX: {str(e)}"
    
    def _process_code(self, filepath: str) -> Tuple[str, None]:
        """Processa file di codice"""
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Aggiungi contesto sul tipo di file
        _, ext = os.path.splitext(filepath)
        return f"[File {ext}]\n\n{content}", None
    
    def save_upload(self, file_data: bytes, filename: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Salva un file caricato
        
        Args:
            file_data: Dati binari del file
            filename: Nome del file
            
        Returns:
            Tupla (percorso_file, messaggio_errore)
        """
        if not self.is_allowed_file(filename):
            return None, "Tipo di file non permesso"
        
        # Crea nome file sicuro
        safe_filename = self._make_safe_filename(filename)
        filepath = os.path.join(self.uploads_dir, safe_filename)
        
        # Evita sovrascritture
        base, ext = os.path.splitext(filepath)
        counter = 1
        while os.path.exists(filepath):
            filepath = f"{base}_{counter}{ext}"
            counter += 1
        
        try:
            with open(filepath, 'wb') as f:
                f.write(file_data)
            return filepath, None
        except Exception as e:
            return None, f"Errore salvataggio file: {str(e)}"
    
    # Nomi riservati Windows (causano errori I/O)
    _RESERVED_NAMES = frozenset({
        'CON', 'PRN', 'AUX', 'NUL',
        *(f'COM{i}' for i in range(1, 10)),
        *(f'LPT{i}' for i in range(1, 10)),
    })

    def _make_safe_filename(self, filename: str) -> str:
        """Rende sicuro un nome file"""
        # Rimuovi null bytes
        safe = filename.replace('\x00', '')
        # Rimuovi caratteri pericolosi
        dangerous_chars = ['/', '\\', ':', '*', '?', '"', '<', '>', '|']
        for char in dangerous_chars:
            safe = safe.replace(char, '_')
        # Limita lunghezza (255 caratteri max per la maggior parte dei filesystem)
        name, ext = os.path.splitext(safe)
        # Blocca nomi riservati Windows
        if name.upper() in self._RESERVED_NAMES:
            name = f"_{name}"
        max_name_len = 255 - len(ext)
        if len(name) > max_name_len:
            name = name[:max_name_len]
        safe = name + ext
        safe = safe.strip('. ')
        # Proteggi da filename vuoto dopo sanitizzazione
        if not safe:
            safe = "_upload"
        return safe
    
    def get_file_info(self, filepath: str) -> dict:
        """
        Ottiene informazioni su un file
        
        Args:
            filepath: Percorso del file
            
        Returns:
            Dizionario con informazioni sul file
        """
        if not os.path.exists(filepath):
            return None
        
        stat = os.stat(filepath)
        return {
            'name': os.path.basename(filepath),
            'size': stat.st_size,
            'size_mb': round(stat.st_size / (1024 * 1024), 2),
            'extension': os.path.splitext(filepath)[1],
            'created': stat.st_ctime,
        }
    
    def clean_old_uploads(self, days: int = 7):
        """
        Pulisce i file caricati più vecchi di X giorni
        
        Args:
            days: Numero di giorni
        """
        current_time = time.time()
        max_age = days * 24 * 60 * 60  # in secondi
        
        removed = 0
        try:
            for filename in os.listdir(self.uploads_dir):
                filepath = os.path.join(self.uploads_dir, filename)
                
                if os.path.isfile(filepath):
                    file_age = current_time - os.path.getmtime(filepath)
                    if file_age > max_age:
                        os.remove(filepath)
                        removed += 1
            
            if removed > 0:
                logger.info("Rimossi %d file upload vecchi", removed)
                
        except Exception as e:
            logger.error("Errore pulizia upload vecchi: %s", e)
