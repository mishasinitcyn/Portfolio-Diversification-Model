import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class ApiService {

  private apiUrl = 'http://localhost:5000';  // URL to the Python server

  constructor(private http: HttpClient) { }

  getStockData(ticker: string, period: string): Observable<any> {
    const params = new HttpParams()
      .set('ticker', ticker)
      .set('period', period);

    return this.http.get<any>(`${this.apiUrl}/stock_data`, { params: params });
  }

  getForecastData(ticker: string, period: string): Observable<any> {
    const params = new HttpParams()
      .set('ticker', ticker)
      .set('period', period);

    return this.http.get<any>(`${this.apiUrl}/forecast`, { params: params });
  }
}