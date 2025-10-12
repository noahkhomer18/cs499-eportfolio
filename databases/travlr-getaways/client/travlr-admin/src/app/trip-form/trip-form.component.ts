import { Component, OnInit } from "@angular/core";
import { ActivatedRoute, Router } from "@angular/router";
import { TripService } from "../services/trip.service";
import { Trip } from "../trip";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";

@Component({
  selector: "app-trip-form",
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: "./trip-form.component.html",
  styleUrls: ["./trip-form.component.css"]
})
export class TripFormComponent implements OnInit {
  trip: Trip = {
    code: "",
    name: "",
    length: "",
    start: new Date(),
    resort: "",
    perPerson: "",
    image: "",
    description: ""
  };
  isEditMode: boolean = false;

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private tripService: TripService
  ) {}

  ngOnInit(): void {
    const tripCode = this.route.snapshot.paramMap.get("tripCode");
    if (tripCode) {
      this.isEditMode = true;
      this.tripService.getTrip(tripCode).subscribe({
        next: (data) => {
          this.trip = data;
        },
        error: (err) => {
          console.error("Error fetching trip for edit", err);
          alert("Failed to load trip for editing.");
          this.router.navigate(["/"]);
        }
      });
    }
  }

  onSubmit(): void {
    if (this.isEditMode) {
      this.tripService.updateTrip(this.trip).subscribe({
        next: () => {
          this.router.navigate(["/trips", this.trip.code]);
        },
        error: (err) => {
          console.error("Error updating trip", err);
          alert("Failed to update trip.");
        }
      });
    } else {
      this.tripService.addTrip(this.trip).subscribe({
        next: () => {
          this.router.navigate(["/"]);
        },
        error: (err) => {
          console.error("Error adding trip", err);
          alert("Failed to add trip.");
        }
      });
    }
  }

  onCancel(): void {
    this.router.navigateByUrl("/");
  }
}




