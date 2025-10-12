# 🔌 API Documentation

## Base URL
```
http://localhost:3000/api
```

## Authentication
All protected routes require a JWT token in the Authorization header:
```
Authorization: Bearer <your-jwt-token>
```

## 📋 Endpoints

### 🔐 Authentication Endpoints

#### Register User
```http
POST /api/register
Content-Type: application/json

{
  "name": "John Doe",
  "email": "john@example.com",
  "password": "your-secure-password"
}
```

**Response:**
```json
{
  "token": "jwt-token-here",
  "user": {
    "id": "user-id",
    "name": "John Doe",
    "email": "john@example.com",
    "role": "user"
  }
}
```

#### Login User
```http
POST /api/login
Content-Type: application/json

{
  "email": "john@example.com",
  "password": "your-secure-password"
}
```

#### Get Current User
```http
GET /api/me
Authorization: Bearer <token>
```

### 🏝️ Trip Endpoints

#### Get All Trips
```http
GET /api/trips?destination=hawaii&minPrice=100&maxPrice=500&sortBy=perPerson&sortOrder=asc
```

**Query Parameters:**
- `destination` - Filter by resort/destination
- `minPrice` - Minimum price filter
- `maxPrice` - Maximum price filter
- `sortBy` - Sort field (perPerson, start, name)
- `sortOrder` - Sort direction (asc, desc)

#### Get Single Trip
```http
GET /api/trips/:code
```

#### Create Trip (Admin Only)
```http
POST /api/trips
Authorization: Bearer <admin-token>
Content-Type: application/json

{
  "code": "HAW001",
  "name": "Hawaii Adventure",
  "length": 7,
  "start": "2024-06-01",
  "resort": "Paradise Resort",
  "perPerson": 1200,
  "image": "hawaii.jpg",
  "description": "Amazing Hawaiian adventure"
}
```

#### Update Trip (Admin Only)
```http
PUT /api/trips/:code
Authorization: Bearer <admin-token>
Content-Type: application/json

{
  "name": "Updated Trip Name",
  "perPerson": 1300
}
```

#### Delete Trip (Admin Only)
```http
DELETE /api/trips/:code
Authorization: Bearer <admin-token>
```

### ⭐ Review Endpoints

#### Get Reviews for Trip
```http
GET /api/trips/:tripId/reviews
```

#### Get User's Reviews
```http
GET /api/reviews/my
Authorization: Bearer <token>
```

#### Create Review
```http
POST /api/reviews
Authorization: Bearer <token>
Content-Type: application/json

{
  "tripId": "trip-object-id",
  "rating": 5,
  "comment": "Amazing trip! Highly recommended."
}
```

#### Update Review
```http
PUT /api/reviews/:id
Authorization: Bearer <token>
Content-Type: application/json

{
  "rating": 4,
  "comment": "Updated review comment"
}
```

#### Delete Review
```http
DELETE /api/reviews/:id
Authorization: Bearer <token>
```

### 📊 Admin Endpoints

#### Get Analytics (Admin Only)
```http
GET /api/admin/stats
Authorization: Bearer <admin-token>
```

**Response:**
```json
{
  "totals": {
    "trips": 25,
    "users": 150,
    "reviews": 89
  },
  "averageRating": 4.2,
  "ratingDistribution": [
    { "_id": 1, "count": 2 },
    { "_id": 2, "count": 5 },
    { "_id": 3, "count": 12 },
    { "_id": 4, "count": 35 },
    { "_id": 5, "count": 35 }
  ],
  "mostReviewedTrips": [
    {
      "tripName": "Hawaii Adventure",
      "tripCode": "HAW001",
      "reviewCount": 15,
      "averageRating": 4.5
    }
  ],
  "recentReviews": [...]
}
```

## 🔒 Error Responses

### 400 Bad Request
```json
{
  "message": "Validation error",
  "errors": [
    {
      "field": "email",
      "message": "Valid email is required"
    }
  ]
}
```

### 401 Unauthorized
```json
{
  "message": "No token, authorization denied"
}
```

### 403 Forbidden
```json
{
  "message": "Access denied. Admin role required."
}
```

### 404 Not Found
```json
{
  "message": "Trip not found"
}
```

### 500 Internal Server Error
```json
{
  "message": "Error fetching trips",
  "error": "Detailed error message"
}
```

## 📝 Data Models

### Trip Model
```json
{
  "_id": "ObjectId",
  "code": "HAW001",
  "name": "Hawaii Adventure",
  "length": 7,
  "start": "2024-06-01T00:00:00.000Z",
  "resort": "Paradise Resort",
  "perPerson": 1200,
  "image": "hawaii.jpg",
  "description": "Amazing Hawaiian adventure",
  "createdAt": "2024-01-01T00:00:00.000Z",
  "updatedAt": "2024-01-01T00:00:00.000Z"
}
```

### Review Model
```json
{
  "_id": "ObjectId",
  "tripId": "ObjectId",
  "userId": "ObjectId",
  "rating": 5,
  "comment": "Amazing trip!",
  "dateCreated": "2024-01-01T00:00:00.000Z",
  "createdAt": "2024-01-01T00:00:00.000Z",
  "updatedAt": "2024-01-01T00:00:00.000Z"
}
```

### User Model
```json
{
  "_id": "ObjectId",
  "name": "John Doe",
  "email": "john@example.com",
  "role": "user",
  "createdAt": "2024-01-01T00:00:00.000Z",
  "updatedAt": "2024-01-01T00:00:00.000Z"
}
```

## 🧪 Testing Examples

### Using cURL

#### Register a new user
```bash
curl -X POST http://localhost:3000/api/register \
  -H "Content-Type: application/json" \
  -d '{"name":"John Doe","email":"john@example.com","password":"your-secure-password"}'
```

#### Get all trips
```bash
curl http://localhost:3000/api/trips
```

#### Create a review (with authentication)
```bash
curl -X POST http://localhost:3000/api/reviews \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -d '{"tripId":"trip-id","rating":5,"comment":"Great trip!"}'
```

### Using Postman

1. Import the API collection
2. Set up environment variables:
   - `base_url`: `http://localhost:3000/api`
   - `jwt_token`: Your JWT token
3. Run the requests in sequence (register → login → use token for protected routes)
