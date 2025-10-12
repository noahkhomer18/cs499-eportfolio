import { Component, Input } from "@angular/core";
import { CommonModule } from "@angular/common";
import { Trip } from "../trip";
import { RouterLink } from "@angular/router";

@Component({
  selector: "app-trip-card",
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: "./trip-card.component.html",
  styleUrls: ["./trip-card.component.css"]
})
export class TripCardComponent {
  @Input() trip!: Trip;
}
