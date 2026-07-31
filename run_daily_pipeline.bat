@echo off
REM Wrapper pro Agendador de Tarefas do Windows chamar o pipeline diário.
REM Ajuste o caminho abaixo se a sua pasta do projeto for diferente.

cd /d C:\Users\mathe\Desktop\ItauQuantAI\momentum_punch
C:\Users\mathe\Miniconda3\python.exe daily_pipeline.py
