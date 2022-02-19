# pull official base image
FROM node:alpine as builder

# set working directory
WORKDIR /app

# add `/app/node_modules/.bin` to $PATH
ENV PATH /app/node_modules/.bin:$PATH

# install app dependencies
COPY package.json ./
COPY package-lock.json ./
# RUN npm install --silent
RUN npm ci --silent
RUN npm install react-scripts@3.4.1 -g --silent

# add app
COPY . /app
# EXPOSE 3000
# start app
RUN ["npm", "run", "build"]

FROM nginx:stable-alpine
COPY --from=builder /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]    