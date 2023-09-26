import { Component, OnInit, Input, OnChanges, SimpleChanges } from '@angular/core';
import { Chart, registerables } from 'chart.js';
import { UCBObject } from 'src/interfaces';
import annotationPlugin from 'chartjs-plugin-annotation';

// Register the plugin
Chart.register(annotationPlugin);
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
        // annotations: this.createAnnotations()
      }
    }
  }; 
  
  /*
  public createAnnotations(): any[] {
    return this.data.map((UCBObject, idx) => {
      const mean = UCBObject.ucb_tuple[1];
      const stdDev = Math.sqrt(UCBObject.ucb_tuple[2]); 
      return {
        type: 'box',
        xScaleID: 'x-axis-0',
        yScaleID: 'y-axis-0',
        xMin: idx - 0.5,  // Assuming bars are 1 unit wide
        xMax: idx + 0.5,
        yMin: mean - stdDev,
        yMax: mean + stdDev,
        backgroundColor: 'rgba(0, 255, 0, 0.1)', 
        borderColor: 'rgba(0, 255, 0, 0.5)',    
        borderWidth: 1
      };
    });
  }
  */

  ngOnInit(): void {
    this.updateChartData();
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['data'] && !changes['data'].isFirstChange()) {
        this.updateChartData();
        this.updateAnnotations();
    }
}

  updateAnnotations(): void {
    // this.chartOptions.plugins.annotation.annotations = this.createAnnotations();
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
