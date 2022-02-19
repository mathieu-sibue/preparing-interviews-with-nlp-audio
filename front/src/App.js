import React from 'react';
import AppContainer from "./components/pages/AppContainer";
import {
    BrowserRouter as Router,
    Switch,
    Route
} from "react-router-dom";
import "./App.css";
import UserContextWrapper from "./contexts/UserContextWrapper";
import HomePage from "./components/pages/HomePage/HomePage";
import TrainingPage from "./components/pages/TrainingPage/TrainingPage";
import Header from "./components/Header";
import TrainingPageContainer from "./components/pages/TrainingPage/TrainingPageContainer";

function App() {
  return (
      <UserContextWrapper>
          <Router>
              <Header/>
              <AppContainer>
                  <Switch>
                      <Route path="/training">
                          <TrainingPageContainer />
                      </Route>
                      <Route path="/">
                          <HomePage />
                      </Route>
                  </Switch>
              </AppContainer>
          </Router>
      </UserContextWrapper>
  );
}

export default App;
