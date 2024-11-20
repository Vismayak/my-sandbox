import React, { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import axios from "axios";

const CreatePost = () => {
    const [title, setTitle] = useState("");
    const [body, setBody] = useState("");

    const queryClient = useQueryClient();

    const postFunction = (newPost) => {
        axios.post("https://jsonplaceholder.typicode.com/posts", newPost);
    }

    const mutation = useMutation({
        mutationFn: postFunction,
        onSuccess: () => {
            queryClient.invalidateQueries({queryKey: ['posts']} );
        }
    })

    const submitPost = () => {
        mutation.mutate();
    }

    if (mutation.isLoading) {
        return <span>Submitting...</span>;
      }
    
      if (mutation.isError) {
        return <span>Error: {mutation.error.message}</span>;
      }
    
      if (mutation.isSuccess) {
        return <span>Post submitted!</span>;
      }

    return(
        <div>
            <h1>Create Post</h1>
            <input type="text" placeholder="Title" value={title} onChange={(e) => setTitle(e.target.value)} />
            <input type="text" placeholder="Body" value={body} onChange={(e) => setBody(e.target.value)} />
            <button onClick={submitPost}>Create Post</button>
        </div>
    )

}

export default CreatePost;