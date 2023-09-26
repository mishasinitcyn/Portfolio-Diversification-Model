export interface UCBObject {
  ucb_tuple: [number, number, number];
  ticker: string;
  forecast_data?: number[][]; // Add this line
}