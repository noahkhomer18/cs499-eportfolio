import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, Router, RouterLink } from "@angular/router";
import { TripService } from "../services/trip.service";
import { Trip } from "../trip";
import { CommonModule } from "@angular/common";
import { AuthService } from "../auth.service";

@Component({
  selector: "app-trip-detail",
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: "./trip-detail.component.html",
  styleUrls: ["./trip-detail.component.css"]
})
export class TripDetailComponent implements OnInit {
  trip: Trip | undefined;
  public userIsLoggedIn: boolean = false; // Add this public property

  constructor(
    private route: ActivatedRoute,
    private tripService: TripService,
    private router: Router,
    private authService: AuthService
  ) {}

  ngOnInit(): void {
    this.userIsLoggedIn = this.authService.isLoggedIn(); // Set the property here
    // console.log("TripDetailComponent ngOnInit: userIsLoggedIn =", this.userIsLoggedIn); // Debug log

    const tripCode = this.route.snapshot.paramMap.get("tripCode");
    if (tripCode) {
      this.tripService.getTrip(tripCode).subscribe({
        next: (data) => {
          this.trip = data;
          // console.log("TripDetailComponent fetched trip:", this.trip); // Debug log
        },
        error: (err) => {
          console.error("Error fetching trip details", err);
        }
      });
    }
  }

  // Remove the isLoggedIn() method as we'll use userIsLoggedIn property directly

  editTrip(): void {
    if (this.trip) {
      this.router.navigate(["/trips/edit", this.trip.code]);
    }
  }

  deleteTrip(): void {
    if (this.trip && confirm("Are you sure you want to delete this trip?")) {
      this.tripService.deleteTrip(this.trip.code).subscribe({
        next: () => {
          this.router.navigate(["/"]);
        },
        error: (err) => {
          console.error("Error deleting trip", err);
          alert("Failed to delete trip.");
        }
      });
    }
  }
}




