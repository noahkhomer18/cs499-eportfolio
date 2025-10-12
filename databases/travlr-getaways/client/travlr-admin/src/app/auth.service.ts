import { Injectable } from "@angular/core";
import { HttpClient } from "@angular/common/http";
import { Router } from "@angular/router";
import { Observable } from "rxjs";
import { User } from "./user";
import { AuthResponse } from "./auth-response";

@Injectable({ providedIn: "root" })
export class AuthService {
  private baseUrl = "http://localhost:3000/api";
  constructor(private http: HttpClient, private router: Router) {}

  private processAuthResponse(response: AuthResponse): void {
    if (response.token) {
      this.saveToken(response.token);
    }
  }

  register(user: User, password: string): void {
    this.http
      .post<AuthResponse>(`${this.baseUrl}/register`, { ...user, password })
      .subscribe({
        next: (resp) => this.processAuthResponse(resp),
        error: (err) => console.error("Registration failed", err)
      });
  }

  login(user: User, password: string): void {
    this.http
      .post<AuthResponse>(`${this.baseUrl}/login`, { ...user, password })
      .subscribe({
        next: (resp) => this.processAuthResponse(resp),
        error: (err) => console.error("Login failed", err)
      });
  }

  saveToken(token: string) { localStorage.setItem("jwt_token", token); }
  getToken(): string | null { return localStorage.getItem("jwt_token"); }
  isLoggedIn(): boolean {
    const token = this.getToken();
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        return payload.exp > (Date.now() / 1000);
      } catch (e) {
        console.error("Invalid token format", e);
        return false;
      }
    }
    return false;
  }

  logout() {
    localStorage.removeItem("jwt_token");
    this.router.navigate(["/login"]);
  }
}
