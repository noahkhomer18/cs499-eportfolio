import { Component, Input, OnInit } from '@angular/core';
import { Review, ReviewService } from '../services/review.service';
import { AuthService } from '../auth.service';

@Component({
  selector: 'app-review',
  templateUrl: './review.component.html',
  styleUrls: ['./review.component.css']
})
export class ReviewComponent implements OnInit {
  @Input() tripId: string = '';
  reviews: Review[] = [];
  newReview: Partial<Review> = {
    tripId: '',
    rating: 5,
    comment: ''
  };
  isLoggedIn: boolean = false;
  showReviewForm: boolean = false;

  constructor(
    private reviewService: ReviewService,
    private authService: AuthService
  ) {}

  ngOnInit() {
    this.isLoggedIn = this.authService.isLoggedIn();
    this.newReview.tripId = this.tripId;
    this.loadReviews();
  }

  loadReviews() {
    if (this.tripId) {
      this.reviewService.getReviewsByTrip(this.tripId).subscribe({
        next: (reviews) => this.reviews = reviews,
        error: (err) => console.error('Error loading reviews:', err)
      });
    }
  }

  submitReview() {
    if (this.newReview.rating && this.newReview.comment) {
      this.reviewService.createReview(this.newReview).subscribe({
        next: (review) => {
          this.reviews.unshift(review);
          this.newReview = { tripId: this.tripId, rating: 5, comment: '' };
          this.showReviewForm = false;
        },
        error: (err) => console.error('Error creating review:', err)
      });
    }
  }

  toggleReviewForm() {
    this.showReviewForm = !this.showReviewForm;
  }

  getStars(rating: number): string[] {
    return Array(5).fill(0).map((_, i) => i < rating ? '★' : '☆');
  }
}
