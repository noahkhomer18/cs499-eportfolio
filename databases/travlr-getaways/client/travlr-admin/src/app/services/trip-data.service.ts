import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

@Injectable({ providedIn: 'root' })
export class TripDataService {
  private apiUrl = '/api/trips';

  constructor(private http: HttpClient) {}

  public getTrips(): Observable<any> {
    return this.http.get(this.apiUrl);
  }

  public addTrip(trip: any): Observable<any> {
    return this.http.post(this.apiUrl, trip);
  }
}
