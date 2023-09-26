import { Component } from '@angular/core';
import { UCBObject } from 'src/interfaces';
import { ApiService } from './api.service';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent {
  title = 'UCB';
  UCBs: UCBObject[] = [];
  tickerInput = '';

  constructor(private apiService: ApiService) { }

  ngOnInit() {
    
  }

  getStock(ticker: string, period: string){
    this.apiService.getStockData(ticker, period).subscribe(
      (data: any) => {
        this.UCBs = [...this.UCBs, data]
      },
      (error) => {
        console.error("Error fetching stock data:", error);
      }
    );
  }

}
