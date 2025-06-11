FROM jenkins/jenkins:lts
EXPOSE 8080
USER root
RUN apt-get update && apt-get install -y python3.11 python3.11-dev python3.11-venv
USER jenkins
