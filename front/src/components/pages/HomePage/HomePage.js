import React from 'react';
import {Button, Container, Grid, Typography} from "@material-ui/core";
import {Redirect, withRouter} from 'react-router-dom';
import TextField from "@material-ui/core/TextField";
import {UserContext} from "../../../contexts/UserContextWrapper";
import remote from "../../../images/remote.svg"
import Box from "@material-ui/core/Box";

function HomePage(props){
    const [username, setUsername] = React.useState(null)
    const {logIn, user} = React.useContext(UserContext)

    return(
        <>
        {(user === null) ?
                <Container style={{textAlign:'center'}}>
                    <Grid container direction="row" alignContent="center" alignItems="center" style={{margin : 20}} spacing={3}>
                        <Grid item xs={6}>
                            <img src={remote} width="130%"/>
                        </Grid>
                        <Grid item xs={6}>
                            <div /*style={{margin : 20}}*/>
                                <Typography variant="h6"><b>Déployez votre meilleur potentiel grâce à :</b></Typography>
                                <Typography variant="h3" color={"primary"} style={{fontFamily:'Grandstander'}}>MyCoach</Typography>
                                <br/>
                                <Box>
                                    <div style={{width: "400px", fontSize:"14px", textAlign:'left', margin: 'auto', color:'grey'}}>
                                        MyCoach est un outil d'entrainement aux entretiens vidéos. À l'aide de modèles de Machine Learning dernier cri et d'une analyse audio approfondie,
                                        MyCoach étudie vos réponses à des questions extraites d'entretiens RH pour vous aider à mieux vous vendre. 
                                    </div>
                                    <br/>
                                    <br/>
                                    <Typography variant="h9">En avant pour une première session d'entrainement !</Typography>
                                </Box>
                                <br/>
                            </div>
                            <div>
                                <TextField onChange={(e) => {setUsername(e.target.value)}} value={username} id="outlined-basic" label="Pseudo" variant="outlined" size="small"/>
                            </div>
                            <div>

                                <Button 
                                    onClick={ () => {logIn(username) ; props.history.push('/training')}} 
                                    style={{marginTop:10, borderRadius:'30px'}} 
                                    color="primary" 
                                    size="large"
                                    variant="contained"
                                >
                                    S'entrainer
                                </Button>
                            </div>
                        </Grid>

                    </Grid>

                </Container>
                :
                <Redirect to={'/training'}/>
        }
        </>
    )
}

export default withRouter(HomePage);