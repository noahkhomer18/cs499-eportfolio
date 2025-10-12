import { Injectable } from "@angular/core";
import { HttpClient, HttpHeaders } from "@angular/common/http";
import { Observable } from "rxjs";
import { Trip } from "../trip";
import { AuthService } from "../auth.service";

@Injectable({ providedIn: "root" })
export class TripService {
  private apiUrl = "http://localhost:3000/api/trips";
  constructor(private http: HttpClient, private authService: AuthService) {}

  private getAuthHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    return new HttpHeaders().set("Authorization", `Bearer ${token}`);
  }

  getTrips(): Observable<Trip[]> {
    return this.http.get<Trip[]>(this.apiUrl);
  }

  getTrip(tripCode: string): Observable<Trip> {
    return this.http.get<Trip>(`${this.apiUrl}/${tripCode}`);
  }

  addTrip(trip: Trip): Observable<Trip> {
    return this.http.post<Trip>(this.apiUrl, trip, { headers: this.getAuthHeaders() });
  }

  updateTrip(trip: Trip): Observable<Trip> {
    return this.http.put<Trip>(`${this.apiUrl}/${trip.code}`, trip, {
      headers: this.getAuthHeaders()
    });
  }

  deleteTrip(tripCode: string): Observable<any> {
    return this.http.delete<any>(`${this.apiUrl}/${tripCode}`, {
      headers: this.getAuthHeaders()
    });
  }
}
