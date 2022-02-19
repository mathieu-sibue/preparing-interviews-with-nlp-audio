import React from 'react';
import {Container} from "@material-ui/core";


function AppContainer(props){
    return(
        <Container maxWidth={"xl"}>
            {props.children}
        </Container>
    )
}

export default AppContainer;