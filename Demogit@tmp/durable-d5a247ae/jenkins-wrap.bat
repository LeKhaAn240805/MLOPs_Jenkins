@echo off 
cmd /c call "D:/MLOps/Jenkins/Demogit@tmp/durable-d5a247ae/jenkins-main.bat" > "D:/MLOps/Jenkins/Demogit@tmp/durable-d5a247ae/jenkins-log.txt" 2>&1
echo %ERRORLEVEL% > "D:/MLOps/Jenkins/Demogit@tmp/durable-d5a247ae/jenkins-result.txt"
