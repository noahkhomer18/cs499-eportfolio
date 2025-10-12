import { Injectable, Provider } from "@angular/core";
import { HttpRequest, HttpHandler, HttpEvent, HttpInterceptor, HTTP_INTERCEPTORS } from "@angular/common/http";
import { Observable } from "rxjs";
import { AuthService } from "../auth.service";

@Injectable()
export class JwtInterceptor implements HttpInterceptor {
  constructor(private authService: AuthService) {}
  intercept(request: HttpRequest<any>, next: HttpHandler): Observable<HttpEvent<any>> {
    const token = this.authService.getToken();
    if (token) {
      request = request.clone({ setHeaders: { Authorization: `Bearer ${token}` } });
    }
    return next.handle(request);
  }
}

export const authInterceptProvider: Provider = {
  provide: HTTP_INTERCEPTORS,
  useClass: JwtInterceptor,
  multi: true
};
