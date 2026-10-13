"""
main.py
Точка входа для запуска настольного приложения распознавания.
"""
from gui import RecognitionApp

if __name__ == "__main__":
    app = RecognitionApp()
    app.mainloop()