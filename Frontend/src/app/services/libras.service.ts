import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Prediction {
  word: string;
  confidence: number;
  rank: number;
}

export interface PredictResponse {
  buffer_progress: number;
  ready: boolean;
  predictions: Prediction[] | null;
  top1_word?: string;
  top1_confidence?: number;
  is_confident?: boolean;
}

export interface ClassesResponse {
  classes: string[];
  num_classes: number;
}

@Injectable({
  providedIn: 'root'
})
export class LibrasService {
  private apiUrl = 'http://localhost:5001';

  constructor(private http: HttpClient) { }

  predict(imageData: string): Observable<PredictResponse> {
    return this.http.post<PredictResponse>(
      `${this.apiUrl}/predict`,
      { image: imageData }
    );
  }

  clearBuffer(): Observable<any> {
    return this.http.post(`${this.apiUrl}/clear`, {});
  }

  getClasses(): Observable<ClassesResponse> {
    return this.http.get<ClassesResponse>(`${this.apiUrl}/classes`);
  }

  predictLetter(imageData: string, targetLetter: string): Observable<any> {
    return this.predict(imageData);
  }

  recordData(imageData: string, label: string): Observable<any> {
    const data = { image: imageData, label: label };
    return this.http.post(`${this.apiUrl}/record_data`, data);
  }
}

