//const burl = "https://#####.mvp.#####.fr"
const burl = "http://localhost:5000"


export default {
    postVideo : async function(blobFile, fileName, question){
        let form = new FormData();
        form.append('file', blobFile, fileName+'.webm');
        form.append("question", question);
        return fetch(burl + '/insights/compute', {
            method: 'POST',
            body: form,
            redirect: 'follow'
        }).then(response => {
        if(response.ok){
            try {
                return response.json(); 
            } catch (error) {
                console.error(error);
            }
        }}
        ).then(response => {
            return response
        })
    }
}
