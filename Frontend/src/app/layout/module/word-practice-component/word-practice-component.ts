import { Component, OnInit, OnDestroy, ViewChild, ElementRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatCardModule } from '@angular/material/card';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressBarModule } from '@angular/material/progress-bar';
import { MatIconModule } from '@angular/material/icon';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { LibrasService, Prediction } from '../../../services/libras.service';

@Component({
  selector: 'app-word-practice',
  standalone: true,
  imports: [
    CommonModule,
    MatCardModule,
    MatButtonModule,
    MatProgressBarModule,
    MatIconModule,
    MatProgressSpinnerModule
  ],
  templateUrl: './word-practice-component.html',
  styleUrls: ['./word-practice-component.scss']
})
export class WordPracticeComponent implements OnInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef;
  @ViewChild('canvasElement') canvasElement!: ElementRef;

  videoStream!: MediaStream;
  videoWidth = 640;
  videoHeight = 480;

  availableWords: string[] = [];
  targetWord: string = '';
  predictions: Prediction[] | null = null;
  bufferProgress = 0;
  isReady = false;
  isCorrect = false;
  isLoading = false;
  feedbackMessage = 'Posicione-se na frente da câmera e faça o sinal.';
  history: string[] = [];
  score = 0;

  processingInterval: any;
  private isProcessing = false;

  constructor(private librasService: LibrasService) {}

  ngOnInit(): void {
    this.loadClasses();
    this.startWebcam();
  }

  ngOnDestroy(): void {
    this.stopWebcam();
    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }
  }

  loadClasses(): void {
    this.librasService.getClasses().subscribe({
      next: (response: any) => {
        this.availableWords = response.classes;
        this.selectRandomWord();
      },
      error: (_err: any) => {
        this.availableWords = ['Sim', 'Oi', 'Obrigado', 'Por favor', 'Bom dia',
                               'Casa', 'Desculpa', 'Trabalho', 'De nada',
                               'Amigo', 'Família'];
        this.selectRandomWord();
      }
    });
  }

  startWebcam(): void {
    navigator.mediaDevices.getUserMedia({ video: true })
      .then(stream => {
        this.videoStream = stream;
        this.videoElement.nativeElement.srcObject = stream;
        this.videoElement.nativeElement.play();
        setTimeout(() => this.startProcessing(), 1000);
      })
      .catch(err => {
        console.error('Erro ao acessar a webcam:', err);
        this.feedbackMessage = 'Erro ao acessar a webcam. Verifique as permissões.';
      });
  }

  stopWebcam(): void {
    if (this.videoStream) {
      this.videoStream.getTracks().forEach(track => track.stop());
    }
  }

  selectRandomWord(): void {
    if (this.availableWords.length === 0) return;
    const randomIndex = Math.floor(Math.random() * this.availableWords.length);
    this.targetWord = this.availableWords[randomIndex];
    this.predictions = null;
    this.isCorrect = false;
    this.isReady = false;
    this.bufferProgress = 0;
    this.feedbackMessage = `Faça o sinal de "${this.targetWord}"`;
    this.librasService.clearBuffer().subscribe();
  }

  startProcessing(): void {
    if (this.processingInterval) clearInterval(this.processingInterval);
    this.processingInterval = setInterval(() => this.sendFrame(), 100);
  }

  sendFrame(): void {
    if (this.isProcessing || this.isCorrect) return;
    if (!this.videoElement?.nativeElement?.srcObject) return;

    this.isProcessing = true;

    const video = this.videoElement.nativeElement;
    const canvas = this.canvasElement.nativeElement;
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageDataUrl = canvas.toDataURL('image/jpeg', 0.7);

    this.librasService.predict(imageDataUrl).subscribe({
      next: (response: any) => {
        this.bufferProgress = response.buffer_progress * 100;
        this.isReady = response.ready;

        if (response.predictions) {
          this.predictions = response.predictions;

          const top1 = response.predictions[0];
          if (top1 && top1.word === this.targetWord && top1.confidence > 0.3) {
            this.isCorrect = true;
            this.score++;
            this.history.push(this.targetWord);
            this.feedbackMessage = `🎉 Correto! Você fez "${this.targetWord}"!`;
            setTimeout(() => this.selectRandomWord(), 2500);
          } else {
            const inTop3 = response.predictions.some((p: any) => p.word === this.targetWord);
            if (inTop3 && response.ready) {
              this.feedbackMessage = `Quase! "${this.targetWord}" está entre as opções.`;
            }
          }
        }

        this.isProcessing = false;
      },
      error: (_err: any) => {
        this.feedbackMessage = 'Erro na conexão com o servidor.';
        this.isProcessing = false;
      }
    });
  }

  clearHistory(): void {
    this.history = [];
    this.score = 0;
    this.selectRandomWord();
  }

  getConfidenceColor(confidence: number): string {
    if (confidence >= 0.7) return '#4caf50';
    if (confidence >= 0.4) return '#ff9800';
    return '#f44336';
  }

  getConfidencePercent(confidence: number): string {
    return (confidence * 100).toFixed(1);
  }
}
