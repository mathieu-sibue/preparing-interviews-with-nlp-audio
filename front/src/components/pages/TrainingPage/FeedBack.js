import React, {useContext, useEffect, useState} from 'react';
import {Container, Typography} from "@material-ui/core";
import CircularProgress from "@material-ui/core/CircularProgress";



function Feedback({feedBackData, children, ...props}){
    const [isLoaded, setIsLoaded] = React.useState(false)

    useEffect(() => {
        setTimeout(()=> setIsLoaded(true),1000)
    })

    return(
        <div style={{marginTop : 10}}>
        { (isLoaded)?
            <><Typography variant="h6" gutterBottom>
                FeedBack
            </Typography>
                <Typography gutterBottom>
                    Ici nous trouverons un beau feedback de la tentative {props.replayingNumber}
                </Typography>
                {children}
            </>
            : <CircularProgress disableShrink style={{margin: "auto"}}/>
        }
        </div>
    )
}

export default Feedback;