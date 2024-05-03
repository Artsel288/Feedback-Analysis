<template>
    <div class="container mt-5 text-center card" style="width: 18rem;">
      <img :src="require(`@/assets/images/user-circle-svgrepo-com.svg`)" class="card-img-top" alt="">
      <div class="card-body">
        <h5 class="card-title">Profile</h5>
      </div>
      <ul class="list-group list-group-flush">
        <li class="list-group-item">ID: {{user.id}}</li>
        <li class="list-group-item">Firstname: {{user.firstname}}</li>
        <li class="list-group-item">Lastname: {{user.lastname}}</li>
        <li class="list-group-item">Email: {{user.email}}</li>
        <li class="list-group-item">Role: {{user.type}}</li>
      </ul>
    </div>

</template>

<script>
import {onMounted, ref} from "vue";
import {useRouter} from "vue-router";
import axiosInstance from "@/axiosInstance";

export default {
  name: "Profile",
  setup() {
    const router = useRouter();
    const user = ref({});


    onMounted(() => {
      axiosInstance.get('/auth/me', {withCredentials: true})
          .then((response) => {
            console.log(response.data)
            user.value = response.data;
            console.log(user.value)
          })
          .catch((error) => {
            console.log(error.response.data)
            router.push('/login')
          })
    });

    return {
      user,
    }
  }
}
</script>
