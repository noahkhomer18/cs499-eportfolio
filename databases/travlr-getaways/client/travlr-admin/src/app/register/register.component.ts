import { Component } from "@angular/core";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { Router } from "@angular/router";
import { AuthService } from "../auth.service";

@Component({
  selector: "app-register",
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: "./register.component.html",
  styleUrls: ["./register.component.css"]
})
export class RegisterComponent {
  credentials = { name: "", email: "", password: "" };

  constructor(private authService: AuthService, private router: Router) {}

  register() {
    const user = { name: this.credentials.name, email: this.credentials.email };
    this.authService.register(user, this.credentials.password);
    setTimeout(() => {
      if (this.authService.isLoggedIn()) {
        this.router.navigateByUrl("/");
      } else {
        alert("Registration failed. Please try again.");
      }
    }, 500);
  }
}
