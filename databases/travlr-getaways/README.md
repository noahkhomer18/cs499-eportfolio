# 🏝️ Travlr Getaways - Full Stack Travel Management System

[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![Angular](https://img.shields.io/badge/Angular-17+-red.svg)](https://angular.io/)
[![MongoDB](https://img.shields.io/badge/MongoDB-6.0+-green.svg)](https://mongodb.com/)
[![Express](https://img.shields.io/badge/Express-5.0+-blue.svg)](https://expressjs.com/)

## 📚 ePortfolio Artifact - CS 499 Capstone Project

This project serves as a **database enhancement artifact** for my **CS 499 Computer Science Capstone Project** at Southern New Hampshire University. The artifact demonstrates advanced database design, implementation, and optimization skills through a comprehensive full-stack travel management system.

### 🎯 Capstone Enhancement Focus
This artifact showcases expertise in **database systems** through:
- **Advanced Schema Design**: Complex relationships between Users, Trips, and Reviews
- **Database Optimization**: Indexing strategies, aggregation pipelines, and query optimization
- **Data Modeling**: One-to-many relationships, referential integrity, and data validation
- **Security Implementation**: Role-based access control, data encryption, and secure authentication
- **Analytics & Reporting**: MongoDB aggregation for business intelligence and statistical analysis

---

## 🎓 CS 465 Full Stack Development Coursework

Originally developed as part of **CS 465 Full Stack Development** coursework, this project demonstrates the evolution from a basic travel application to a sophisticated, production-ready system. The enhancement process showcases the application of advanced database concepts learned throughout the Computer Science program.

### 📈 Enhancement Journey
- **Initial State**:** Basic CRUD operations with simple data storage
- **Enhanced State**:** Advanced database architecture with relationships, security, and analytics
- **Professional Standards**:** Production-ready code with comprehensive documentation and testing

---

A comprehensive full-stack travel management application built with **Express.js**, **Angular**, and **MongoDB**. Features include trip management, user reviews, role-based access control, and advanced analytics.

## 🚀 Features

### ✨ Core Features
- **Trip Management**: Create, read, update, and delete travel packages
- **User Reviews**: Rate and review trips with comments
- **Role-Based Access Control**: Admin and user roles with different permissions
- **Advanced Filtering**: Filter trips by destination, price, and date
- **Analytics Dashboard**: Admin statistics with aggregation pipelines
- **JWT Authentication**: Secure user authentication and authorization

### 🎯 Technical Highlights
- **Backend**: Express.js with MongoDB and Mongoose ODM
- **Frontend**: Angular 17+ with TypeScript
- **Database**: MongoDB with proper schema design and relationships
- **Security**: JWT tokens, password hashing, input validation
- **API Design**: RESTful endpoints with proper HTTP methods

## 📁 Project Structure

```
travlr-getaways/
├── 📁 app_server/                 # Backend Express.js Application
│   ├── 📁 controllers/            # Business logic controllers
│   │   ├── adminController.js     # Admin analytics and stats
│   │   ├── authController.js      # Authentication (login/register)
│   │   ├── reviewController.js     # Review CRUD operations
│   │   ├── tripController.js      # Trip CRUD operations
│   │   └── traveller.js           # Legacy view controller
│   ├── 📁 middleware/             # Custom middleware
│   │   └── auth.js                # JWT authentication & RBAC
│   ├── 📁 models/                 # MongoDB schemas
│   │   ├── Review.js              # Review model with trip reference
│   │   ├── Trip.js                # Trip model with validation
│   │   └── User.js                # User model with roles
│   ├── 📁 routes/                 # API route definitions
│   │   ├── api.js                # Main API routes
│   │   └── traveller.js           # Legacy view routes
│   ├── 📁 utils/                  # Utility functions
│   └── 📁 views/                   # Handlebars templates
├── 📁 client/                     # Frontend Angular Application
│   └── 📁 travlr-admin/           # Angular SPA
│       ├── 📁 src/app/            # Angular components and services
│       │   ├── 📁 review/         # Review components
│       │   ├── 📁 services/       # Angular services
│       │   ├── 📁 trip-*/         # Trip-related components
│       │   └── 📁 auth/            # Authentication components
│       └── 📁 src/assets/          # Static assets and images
├── 📁 docs/                       # Documentation and assets
│   ├── 📁 screenshots/            # Application screenshots
│   └── 📁 assets/                 # Project assets
├── 📁 config/                     # Configuration files
├── 📁 tests/                      # Test files
├── 📄 app.js                      # Main Express application
├── 📄 package.json                # Dependencies and scripts
└── 📄 .env.example               # Environment variables template
```

## 🛠️ Installation & Setup

### Prerequisites
- **Node.js** (v18 or higher)
- **MongoDB** (v6.0 or higher)
- **Angular CLI** (v17 or higher)

### Backend Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/noahkhomer18/CS-465.git
   cd CS-465
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Environment configuration**
   ```bash
   cp .env.example .env
   # Edit .env with your MongoDB URI and JWT secret
   ```

4. **Start MongoDB**
   ```bash
   mongod
   ```

5. **Run the backend server**
   ```bash
   npm start
   # Server runs on http://localhost:3000
   ```

### Frontend Setup

1. **Navigate to Angular app**
   ```bash
   cd client/travlr-admin
   ```

2. **Install Angular dependencies**
   ```bash
   npm install
   ```

3. **Start Angular development server**
   ```bash
   ng serve
   # App runs on http://localhost:4200
   ```

## 🔧 API Endpoints

### Authentication
- `POST /api/register` - User registration
- `POST /api/login` - User login
- `GET /api/me` - Get current user

### Trips
- `GET /api/trips` - Get all trips (with filtering)
- `GET /api/trips/:code` - Get single trip
- `POST /api/trips` - Create trip (admin only)
- `PUT /api/trips/:code` - Update trip (admin only)
- `DELETE /api/trips/:code` - Delete trip (admin only)

### Reviews
- `GET /api/trips/:tripId/reviews` - Get reviews for a trip
- `GET /api/reviews/my` - Get user's reviews
- `POST /api/reviews` - Create review
- `PUT /api/reviews/:id` - Update review
- `DELETE /api/reviews/:id` - Delete review

### Admin
- `GET /api/admin/stats` - Get analytics (admin only)

## 🎨 Features in Detail

### 🔐 Role-Based Access Control (RBAC)
- **Admin Role**: Full CRUD access to trips, view analytics
- **User Role**: View trips, create/edit own reviews
- **JWT Authentication**: Secure token-based authentication

### 📊 Advanced Query Features
- **Trip Filtering**: By destination, price range, date
- **Sorting**: By price, rating, start date
- **Aggregation**: Admin statistics with MongoDB aggregation pipelines

### ⭐ Review System
- **Rating System**: 1-5 star ratings
- **Comments**: Text reviews with validation
- **User Reviews**: One review per user per trip
- **Review Analytics**: Average ratings, review counts

## 🧪 Testing

### API Testing
Use tools like **Postman** or **Insomnia** to test the API endpoints:

```bash
# Example: Get all trips
GET http://localhost:3000/api/trips

# Example: Create a review (requires authentication)
POST http://localhost:3000/api/reviews
Authorization: Bearer <your-jwt-token>
Content-Type: application/json

{
  "tripId": "trip-object-id",
  "rating": 5,
  "comment": "Amazing trip!"
}
```

## 📸 Screenshots

Check the `docs/screenshots/` folder for application screenshots showing:
- Trip listing and details
- Review system interface
- Admin dashboard
- Authentication flows

## 🚀 Deployment

### Environment Variables
Create a `.env` file with:
```env
MONGODB_URI=mongodb://127.0.0.1:27017/travlr
JWT_SECRET=your-super-secret-jwt-key
JWT_EXPIRE=7d
PORT=3000
NODE_ENV=development
```

### Production Considerations
- Use environment-specific MongoDB URIs
- Set strong JWT secrets
- Configure CORS for production domains
- Use HTTPS in production
- Set up proper logging and monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🎓 CS 499 Learning Outcomes Demonstrated

### 🗄️ Database Systems Expertise
This artifact demonstrates mastery of database concepts through:

**Advanced Schema Design**
- Complex entity relationships (Users ↔ Trips ↔ Reviews)
- Referential integrity with foreign key constraints
- Data validation and business rule enforcement
- Optimized indexing strategies for performance

**Database Security & Access Control**
- Role-based access control (RBAC) implementation
- Secure authentication with JWT tokens
- Password hashing and encryption
- Input validation and SQL injection prevention

**Analytics & Business Intelligence**
- MongoDB aggregation pipelines for complex queries
- Statistical analysis of user behavior and preferences
- Performance metrics and system monitoring
- Data-driven decision making capabilities

**Professional Development Practices**
- Comprehensive API documentation
- Code organization and maintainability
- Testing strategies and quality assurance
- Version control and collaborative development

### 🔧 Technical Skills Showcased
- **Database Design**: Entity-Relationship modeling, normalization, indexing
- **Backend Development**: RESTful API design, middleware implementation, security
- **Frontend Integration**: Angular services, reactive programming, user experience
- **System Architecture**: Full-stack integration, scalability considerations
- **Documentation**: Professional-grade technical writing and code documentation

## 📚 Technologies Used

### Backend
- **Express.js** - Web framework
- **MongoDB** - NoSQL database
- **Mongoose** - ODM for MongoDB
- **JWT** - Authentication tokens
- **bcryptjs** - Password hashing
- **express-validator** - Input validation

### Frontend
- **Angular 17+** - Frontend framework
- **TypeScript** - Type-safe JavaScript
- **RxJS** - Reactive programming
- **Bootstrap** - CSS framework

## 📄 License

This project is part of the **CS-465 Full Stack Development** curriculum at **Southern New Hampshire University**.

## 🙏 Acknowledgments

- **Southern New Hampshire University (SNHU)**
- **Professor Paul Davis** for guidance and support
- **CS-465 Full Stack Development** course curriculum

---

## 📞 Support

For questions or support, please contact:
- **GitHub Issues**: [Create an issue](https://github.com/noahkhomer18/CS-465/issues)
- **Email**: [Your contact email]

---

**Built with ❤️ for CS-465 Full Stack Development**