import { ApplicationConfig, importProvidersFrom } from "@angular/core";
import { provideRouter } from "@angular/router";
import { routes } from "./app.routes";
import { provideHttpClient, withInterceptorsFromDi } from "@angular/common/http";
import { FormsModule, ReactiveFormsModule } from "@angular/forms";
import { authInterceptProvider } from "./utils/jwt.interceptor";

export const appConfig: ApplicationConfig = {
  providers: [
    provideRouter(routes),
    provideHttpClient(withInterceptorsFromDi()),
    importProvidersFrom(FormsModule, ReactiveFormsModule),
    authInterceptProvider
  ]
};
