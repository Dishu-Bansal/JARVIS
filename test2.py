import pygetwindow as gw
import psutil
import os
import docx
from PyPDF2 import PdfReader

# --- File Reading Functions (from previous answer) ---
def read_docx_text(file_path):
    """Reads and returns all text from a .docx file."""
    try:
        doc = docx.Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    except Exception as e:
        return f"❌ Error reading DOCX file: {e}"

def read_pdf_text(file_path):
    """Reads and returns all text from a .pdf file."""
    try:
        reader = PdfReader(file_path)
        return '\n'.join([page.extract_text() for page in reader.pages])
    except Exception as e:
        return f"❌ Error reading PDF file: {e}"

def read_txt_file(file_path):
    """Reads and returns all text from a .txt file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"❌ Error reading TXT file: {e}"

# --- Main Application Logic ---
def main():
    """Lists windows, finds the associated file path via PID, and reads the content."""
    print("🔎 Searching for open application windows...")
    
    # We use getAllWindows() to get Window objects, not just titles
    wins = gw.getAllWindows()
    open_windows = [win for win in gw.getAllWindows() if win.title]
    if not open_windows:
        print("No open application windows found.")
        return

    print("\n📝 Please select an application window to read from:")
    for i, window in enumerate(open_windows):
        print(f"  [{i}] {window.title}")

    try:
        # 1. GET USER SELECTION
        choice = int(input("\nEnter the number of your choice: "))
        selected_window = open_windows[choice]
        print(f"\nYou selected: '{selected_window.title}'")

        # 2. GET PROCESS ID (PID) FROM THE WINDOW
        pid = selected_window
        print(f"⚙️ Found Process ID: {pid}")

        # 3. USE PID TO FIND THE OPEN FILE PATH
        proc = psutil.Process(pid)
        
        # Filter open_files for document types
        document_path = None
        open_files = proc.open_files()
        for file in open_files:
            if file.path.endswith(('.docx', '.pdf', '.txt')):
                document_path = file.path
                print(f"📄 Found document file: {document_path}")
                break # Stop after finding the first relevant document

        if not document_path:
            print("\n❌ Could not find an open .docx, .pdf, or .txt file for this process.")
            print("   The application might not have a file open, or it's an unsupported type.")
            return

        # 4. READ CONTENT FROM THE FOUND PATH
        print("\n📖 Reading file content...\n" + "="*25)
        content = ""
        if document_path.endswith('.docx'):
            content = read_docx_text(document_path)
        elif document_path.endswith('.pdf'):
            content = read_pdf_text(document_path)
        elif document_path.endswith('.txt'):
            content = read_txt_file(document_path)
        
        print(content)
        print("="*25 + "\n✅ End of content.")

    except IndexError:
        print("Invalid selection.")
    except ValueError:
        print("Invalid input. Please enter a number.")
    except psutil.NoSuchProcess:
        print(f"Process with PID {pid} no longer exists. The application may have been closed.")
    except psutil.AccessDenied:
        print(f"Access Denied. Cannot inspect process with PID {pid}. Try running the script as an administrator.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    if os.name != 'nt':
        print("This script uses Windows-specific features and is not compatible with your OS.")
    else:
        main()