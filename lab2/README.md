# Для запуска Jenkins Pipeline необходимо:
1. Установить плагин Docker pipeline в Jenkins
2. Создать item --> Создать pipeline
3. В настройках пайплайна нужно указать несколько обязательных настроек:
    - в поле Definition указать "Pipeline script from SCM"
    - SCM выбрать git
    - Ветку репозитория выбрать main
    - В параметре  "Script Path" указать lab2/Jenkinsfile
4. Сохраняем параметры и делаим билд
