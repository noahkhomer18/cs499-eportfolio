import { Component } from "@angular/core";
import { CommonModule } from "@angular/common";
import { FormsModule } from "@angular/forms";
import { Router } from "@angular/router";
import { AuthService } from "../auth.service";

@Component({
  selector: "app-login",
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: "./login.component.html",
  styleUrls: ["./login.component.css"]
})
export class LoginComponent {
  credentials = { email: "", password: "" };

  constructor(private authService: AuthService, private router: Router) {}

  login() {
    const user = { email: this.credentials.email, name: "" };
    this.authService.login(user, this.credentials.password);
    setTimeout(() => {
      if (this.authService.isLoggedIn()) {
        this.router.navigateByUrl("/");
      } else {
        alert("Login failed. Please check your credentials.");
      }
    }, 500);
  }
}
