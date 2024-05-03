# Use an official Node runtime as a parent image
FROM node:20-alpine as build

# Set the working directory in the container
WORKDIR /app

# Copy package.json and package-lock.json to the working directory
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the application code to the working directory
COPY . .

# Build the Vue.js application for production with minification
RUN npm run build

# Use NGINX as the production server
FROM nginx:alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Copy the built Vue.js files from the build stage to NGINX's default public directory
COPY --from=build /app/dist /usr/share/nginx/html

# Expose port 80 to the outside world
EXPOSE 80

# Command to run NGINX in the foreground
RUN /docker-entrypoint.sh $@

CMD ["nginx", "-g", "daemon off;"]
