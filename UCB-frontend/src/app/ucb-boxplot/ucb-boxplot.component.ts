import { Component, OnInit, Input, OnChanges } from '@angular/core';
import { Chart, registerables } from 'chart.js';
import { UCBObject } from 'src/interfaces';

@Component({
  selector: 'app-ucb-boxplot',
  templateUrl: './ucb-boxplot.component.html',
})
export class UcbBoxplotComponent implements OnInit, OnChanges {
  @Input() data: UCBObject[] = []; // An array of tuples (UCB score, mean, variance)

  public chartData: number[][] = [];
  public chartLabels!: string[];
  public chartOptions = {
    responsive: true,
    plugins: {
      annotation: {
        annotations: this.createAnnotations()
      }
    }
  };
  
  public createAnnotations(): any[] {
    return this.data?.map(UCBObject => {
      const mean = UCBObject.ucb_tuple[1]; // Extracting the mean from the tuple
      return {
        type: 'line',
        mode: 'horizontal',
        scaleID: 'y-axis-0',
        value: mean,
        borderColor: 'rgba(255,0,0,0.5)',
        borderWidth: 2,
        label: {
          enabled: true,
          content: `Mean: ${mean}`
        }
      };
    });
  }

  ngOnInit(): void {
    this.updateChartData();
  }

  ngOnChanges(): void {
    this.updateChartData();
    console.log(this.chartData)
  }

  private updateChartData(): void {
    this.chartData = this.data.map(UCBObject => this.calculateBoxPlotValues(UCBObject.ucb_tuple));
    this.chartLabels = this.data.map(UCBObject => UCBObject.ticker);
}

  private calculateBoxPlotValues(tuple: [number, number, number]): number[] {
    const [ucbScore, mean, variance] = tuple;
    const stdDev = Math.sqrt(variance);

    // Calculate the box plot values
    return [
        mean - 2 * stdDev, // Q1
        mean - stdDev,     // Q2
        mean,              // Q3
        mean + stdDev,     // Q4
        mean + 2 * stdDev  // Q5
    ];
  }
}
