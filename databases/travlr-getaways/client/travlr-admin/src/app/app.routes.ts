import { Routes } from "@angular/router";
import { TripListingComponent } from "./trip-listing/trip-listing.component";
import { LoginComponent } from "./login/login.component";
import { RegisterComponent } from "./register/register.component";
import { AuthGuard } from "./auth.guard";
import { TripDetailComponent } from "./trip-detail/trip-detail.component";
import { TripFormComponent } from "./trip-form/trip-form.component";

export const routes: Routes = [
  { path: "", component: TripListingComponent, canActivate: [AuthGuard] },
  { path: "login", component: LoginComponent },
  { path: "register", component: RegisterComponent },
  { path: "trips/add", component: TripFormComponent, canActivate: [AuthGuard] },
  { path: "trips/edit/:tripCode", component: TripFormComponent, canActivate: [AuthGuard] },
  { path: "trips/:tripCode", component: TripDetailComponent, canActivate: [AuthGuard] },
  { path: "**", redirectTo: "" } // Redirect any unknown routes to the home page
];
