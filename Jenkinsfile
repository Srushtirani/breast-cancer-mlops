pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Check Tools') {
            steps {
                sh '''
                    python3 --version
                    git --version
                    docker --version
                '''
            }
        }

        stage('Install Dependencies') {
            steps {
                sh '''
                    python3 -m pip install -r requirements.txt
                '''
            }
        }

        stage('Train Model') {
            steps {
                sh 'python3 train.py'
            }
        }

        stage('Run Pytest') {
            steps {
                sh 'python3 -m pytest -v'
            }
        }

        stage('Generate Drift Report') {
            steps {
                sh 'python3 monitor.py'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                    docker build \
                      -t breast-cancer-mlops:1.0 .
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts(
                artifacts: 'reports/*.html',
                allowEmptyArchive: true
            )
        }

        success {
            echo 'UAT pipeline completed successfully.'
        }

        failure {
            echo 'UAT pipeline failed. Check the console log.'
        }
    }
}