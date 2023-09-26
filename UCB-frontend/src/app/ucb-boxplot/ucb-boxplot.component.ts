import { Component, OnInit, Input, OnChanges } from '@angular/core';

@Component({
  selector: 'app-ucb-boxplot',
  templateUrl: './ucb-boxplot.component.html',
})
export class UcbBoxplotComponent implements OnInit, OnChanges {
  @Input() data!: [number, number, number][]; // An array of tuples (UCB score, mean, variance)

  public chartData: number[][] = [];
  public chartLabels!: string[];
  public chartOptions!: string[];

  ngOnInit(): void {
    this.updateChartData();
  }

  ngOnChanges(): void {
    this.updateChartData();
    console.log(this.chartData)
  }

  private updateChartData(): void {
    this.chartData = this.data.map(tuple => this.calculateBoxPlotValues(tuple));
    this.chartLabels = this.data.map((_, idx) => `Dataset ${idx + 1}`);
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
