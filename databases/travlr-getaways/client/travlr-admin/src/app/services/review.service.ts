import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AuthService } from '../auth.service';

export interface Review {
  _id?: string;
  tripId: string;
  userId: string;
  rating: number;
  comment: string;
  dateCreated: Date;
  user?: {
    name: string;
  };
}

@Injectable({
  providedIn: 'root'
})
export class ReviewService {
  private apiUrl = 'http://localhost:3000/api';

  constructor(private http: HttpClient, private authService: AuthService) {}

  private getAuthHeaders(): HttpHeaders {
    const token = this.authService.getToken();
    return new HttpHeaders().set('Authorization', `Bearer ${token}`);
  }

  getReviewsByTrip(tripId: string): Observable<Review[]> {
    return this.http.get<Review[]>(`${this.apiUrl}/trips/${tripId}/reviews`);
  }

  getMyReviews(): Observable<Review[]> {
    return this.http.get<Review[]>(`${this.apiUrl}/reviews/my`, { 
      headers: this.getAuthHeaders() 
    });
  }

  createReview(review: Partial<Review>): Observable<Review> {
    return this.http.post<Review>(`${this.apiUrl}/reviews`, review, { 
      headers: this.getAuthHeaders() 
    });
  }

  updateReview(id: string, review: Partial<Review>): Observable<Review> {
    return this.http.put<Review>(`${this.apiUrl}/reviews/${id}`, review, { 
      headers: this.getAuthHeaders() 
    });
  }

  deleteReview(id: string): Observable<any> {
    return this.http.delete(`${this.apiUrl}/reviews/${id}`, { 
      headers: this.getAuthHeaders() 
    });
  }
}
