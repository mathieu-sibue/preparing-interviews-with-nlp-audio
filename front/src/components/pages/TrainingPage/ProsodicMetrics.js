import React, { useEffect, useState } from "react";
import { Typography, Paper } from "@material-ui/core";
import wind from "../../../images/wind.svg"
import pause from "../../../images/pause.svg"
import discount from "../../../images/discount.svg";


export default function ProsodicMetrics(props) {

    const [metrics, setMetrics] = useState({
        balance: props.prosodicInsights.balance,
        articulation_rate: props.prosodicInsights.articulation_rate,
        number_of_pauses: props.prosodicInsights.number_of_pauses,
        original_duration: props.prosodicInsights.original_duration
    })

    useEffect(()=>{
        setMetrics({
            balance: props.prosodicInsights.balance,
            articulation_rate: props.prosodicInsights.articulation_rate,
            number_of_pauses: props.prosodicInsights.number_of_pauses,
            original_duration: props.prosodicInsights.original_duration
        })
    }, [props.prosodicInsights])
    
    return (
        <Paper style={{margin: 'auto', padding: '18px', paddingTop: '12px'}} variant='outlined'>
            <Typography align='left' variant="h6" component="h6">
                Diction
            </Typography>
            <div style={{textAlign: 'justify', fontSize:'13px'}}>
                <img src={discount} className="MyLogos"/>&nbsp; Temps de parole / temps total : <b>{Math.round(metrics.balance*10)/10}</b>.
                <br/>
                <img src={wind} className="MyLogos"/>&nbsp; Débit sur le temps de parole : <b>{Math.round(metrics.articulation_rate*100)/100}</b> syllabe(s) par seconde.
                <br/>
                <img src={pause} className="MyLogos"/>&nbsp; Nombre de pauses sur le temps total : <b>{metrics.number_of_pauses}</b> pause(s) en {Math.round(metrics.original_duration*10)/10} secondes.
            </div>
        </Paper>
    )
}