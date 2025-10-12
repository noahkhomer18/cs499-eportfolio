# 🛠️ Development Guide

## 🏗️ Architecture Overview

### Backend Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Angular SPA   │◄──►│   Express API   │◄──►│    MongoDB      │
│   (Frontend)    │    │   (Backend)     │    │   (Database)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack
- **Frontend**: Angular 17+ with TypeScript
- **Backend**: Express.js with Node.js
- **Database**: MongoDB with Mongoose ODM
- **Authentication**: JWT tokens
- **Validation**: express-validator

## 🔧 Development Setup

### Prerequisites
```bash
# Node.js (v18+)
node --version

# MongoDB (v6.0+)
mongod --version

# Angular CLI (v17+)
ng version
```

### Environment Setup

1. **Clone and install**
   ```bash
   git clone https://github.com/noahkhomer18/CS-465.git
   cd CS-465
   npm install
   ```

2. **Backend environment**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Frontend environment**
   ```bash
   cd client/travlr-admin
   npm install
   ```

## 🚀 Running the Application

### Development Mode

1. **Start MongoDB**
   ```bash
   mongod
   ```

2. **Start Backend (Terminal 1)**
   ```bash
   npm run dev
   # Runs on http://localhost:3000
   ```

3. **Start Frontend (Terminal 2)**
   ```bash
   cd client/travlr-admin
   ng serve
   # Runs on http://localhost:4200
   ```

### Production Mode

1. **Build Angular app**
   ```bash
   cd client/travlr-admin
   ng build --prod
   ```

2. **Start production server**
   ```bash
   npm start
   ```

## 📁 Code Organization

### Backend Structure
```
app_server/
├── controllers/          # Business logic
│   ├── authController.js     # Authentication logic
│   ├── tripController.js     # Trip CRUD operations
│   ├── reviewController.js   # Review operations
│   └── adminController.js    # Admin analytics
├── middleware/           # Custom middleware
│   └── auth.js              # JWT & RBAC middleware
├── models/              # Database schemas
│   ├── User.js              # User model with roles
│   ├── Trip.js              # Trip model
│   └── Review.js            # Review model
├── routes/              # API routes
│   ├── api.js               # Main API routes
│   └── traveller.js         # Legacy routes
└── utils/               # Utility functions
```

### Frontend Structure
```
client/travlr-admin/src/app/
├── services/            # Angular services
│   ├── auth.service.ts      # Authentication service
│   ├── trip.service.ts      # Trip operations
│   └── review.service.ts    # Review operations
├── components/          # Angular components
│   ├── trip-*/              # Trip components
│   ├── review/              # Review components
│   └── auth/                # Auth components
└── utils/               # Utility functions
```

## 🔐 Authentication Flow

### JWT Token Flow
```
1. User registers/logs in
2. Server validates credentials
3. Server generates JWT token
4. Client stores token in localStorage
5. Client sends token in Authorization header
6. Server validates token on protected routes
```

### Role-Based Access Control
```javascript
// Middleware usage
router.post('/trips', auth, requireAdmin, tripController.createTrip);
router.post('/reviews', auth, reviewController.createReview);
```

## 🗄️ Database Design

### Collections and Relationships
```
Users (1) ──► (Many) Reviews
Trips (1) ──► (Many) Reviews

Users:
├── _id: ObjectId
├── name: String
├── email: String (unique)
├── password: String (hashed)
└── role: String (admin|user)

Trips:
├── _id: ObjectId
├── code: String (unique)
├── name: String
├── length: Number
├── start: Date
├── resort: String
├── perPerson: Number
├── image: String
└── description: String

Reviews:
├── _id: ObjectId
├── tripId: ObjectId (ref: Trip)
├── userId: ObjectId (ref: User)
├── rating: Number (1-5)
├── comment: String
└── dateCreated: Date
```

## 🧪 Testing

### API Testing with Postman

1. **Import collection**
   - Create new collection in Postman
   - Add environment variables
   - Import API endpoints

2. **Test flow**
   ```
   Register → Login → Get Token → Use Token for Protected Routes
   ```

### Unit Testing
```bash
# Backend tests
npm test

# Frontend tests
cd client/travlr-admin
ng test
```

## 🐛 Debugging

### Common Issues

1. **MongoDB Connection**
   ```bash
   # Check if MongoDB is running
   mongod --version
   
   # Start MongoDB
   mongod
   ```

2. **CORS Issues**
   ```javascript
   // In app.js
   app.use(cors({
     origin: 'http://localhost:4200',
     credentials: true
   }));
   ```

3. **JWT Token Issues**
   ```javascript
   // Check token format
   const token = req.header('Authorization')?.replace('Bearer ', '');
   ```

### Logging
```javascript
// Add logging middleware
app.use((req, res, next) => {
  console.log(`${req.method} ${req.path} - ${new Date().toISOString()}`);
  next();
});
```

## 📦 Dependencies

### Backend Dependencies
```json
{
  "express": "^5.1.0",
  "mongoose": "^8.0.0",
  "cors": "^2.8.5",
  "dotenv": "^16.3.1",
  "bcryptjs": "^2.4.3",
  "jsonwebtoken": "^9.0.2",
  "express-validator": "^7.0.1"
}
```

### Frontend Dependencies
```json
{
  "@angular/core": "^17.0.0",
  "@angular/common": "^17.0.0",
  "@angular/router": "^17.0.0",
  "rxjs": "^7.8.0"
}
```

## 🚀 Deployment

### Environment Variables
```env
# Production
NODE_ENV=production
MONGODB_URI=your-mongodb-connection-string-here
JWT_SECRET=your-super-secret-jwt-key-here
PORT=3000
```

### Build Process
```bash
# Build Angular app
cd client/travlr-admin
ng build --prod

# Start production server
npm start
```

## 📝 Code Style

### Backend (JavaScript)
```javascript
// Use async/await
const getTrips = async (req, res) => {
  try {
    const trips = await Trip.find(query);
    res.json(trips);
  } catch (error) {
    res.status(500).json({ message: 'Error fetching trips' });
  }
};
```

### Frontend (TypeScript)
```typescript
// Use interfaces
export interface Trip {
  _id?: string;
  code: string;
  name: string;
  // ...
}

// Use services for API calls
@Injectable()
export class TripService {
  getTrips(): Observable<Trip[]> {
    return this.http.get<Trip[]>(this.apiUrl);
  }
}
```

## 🔄 Git Workflow

### Branch Strategy
```bash
# Feature development
git checkout -b feature/new-feature
git commit -m "feat: add new feature"
git push origin feature/new-feature

# Bug fixes
git checkout -b fix/bug-description
git commit -m "fix: resolve bug"
```

### Commit Messages
```
feat: add new feature
fix: resolve bug
docs: update documentation
style: code formatting
refactor: code restructuring
test: add tests
chore: maintenance tasks
```

## 📚 Resources

- [Express.js Documentation](https://expressjs.com/)
- [Angular Documentation](https://angular.io/docs)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [JWT.io](https://jwt.io/) - JWT token debugger
- [Postman](https://www.postman.com/) - API testing
