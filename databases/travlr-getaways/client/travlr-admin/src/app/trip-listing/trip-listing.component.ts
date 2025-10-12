import { Component, OnInit } from "@angular/core";
import { CommonModule } from "@angular/common";
import { Trip } from "../trip";
import { TripCardComponent } from "../trip-card/trip-card.component";
import { TripService } from "../services/trip.service";
import { AuthService } from "../auth.service";

@Component({
  selector: "app-trip-listing",
  standalone: true,
  imports: [CommonModule, TripCardComponent],
  templateUrl: "./trip-listing.component.html",
  styleUrls: ["./trip-listing.component.css"]
})
export class TripListingComponent implements OnInit {
  trips: Trip[] = [];
  message = "";

  constructor(private tripService: TripService, private authService: AuthService) {}

  ngOnInit(): void {
    this.tripService.getTrips().subscribe({
      next: (data) => {
        this.trips = (data ?? []) as Trip[];
        this.message = this.trips.length
          ? `There are ${this.trips.length} trips available.`
          : "No trips found. Please add some.";
      },
      error: (err) => {
        console.error("Error fetching trips", err);
        this.message = "Failed to load trips.";
      }
    });
  }

  isLoggedIn(): boolean {
    return this.authService.isLoggedIn();
  }

  logout(): void {
    this.authService.logout();
  }
}
