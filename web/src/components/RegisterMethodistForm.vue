<template>
  <h1 class="text-center h3 mb-3 fw-normal">Register Methodist</h1>

  <form @submit.prevent="submit">
    <div class="form-floating">
      <input class="form-control" :class="{'is-invalid': !isValidFirstname}" name="firstname" placeholder="Name">
      <label>First Name</label>
    </div>

    <div class="form-floating">
      <input class="form-control" :class="{'is-invalid': !isValidLastname }" name="lastname" placeholder="Name">
      <label>Last Name</label>
    </div>

    <div class="form-floating">
      <input type="email" class="form-control" :class="{'is-invalid': !isValidEmail }" name="email"
             placeholder="name@example.com">
      <label>Email address</label>
    </div>

    <div class="form-floating">
      <input type="password" class="form-control" :class="{'is-invalid': !isValidPassword }" name="password"
             placeholder="Password">
      <label>Password</label>
    </div>

    <div v-if="!isValidForm">
      <p style="margin-top: 10px; color: #dc3545; text-align: justify">{{formError}}</p>
    </div>

    <button class="w-100 btn btn-lg btn-primary" type="submit">Submit</button>
  </form>

</template>

<script>
import {ref} from "vue";
import router from "@/router";
import axiosInstance from "@/axiosInstance";

export default {
  name: "RegisterMethodistForm",
  setup() {
    const formError = ref('');
    const isValidForm = ref(true);
    const isValidFirstname = ref(true);
    const isValidLastname = ref(true);
    const isValidEmail = ref(true);
    const isValidPassword = ref(true);

    const resetForm = () =>{
      formError.value = ''
      isValidForm.value = true
      isValidFirstname.value = true
      isValidLastname.value = true
      isValidEmail.value = true
      isValidPassword.value = true
    }

    const submit = (e) => {
      resetForm()
      const form = new FormData(e.target);

      const inputs = JSON.stringify(Object.fromEntries(form.entries()));

      axiosInstance.post('/methodists', inputs)
          .then(() => {
            router.push('/login')
          })
          .catch((error) => {
            switch (error.response.status) {
              case 409:
                isValidForm.value = false;
                isValidEmail.value = false;
                formError.value = 'Methodist with this email already exists';

                break;

              case 422:
                isValidForm.value = false;
                isValidEmail.value = false;
                formError.value = error.response.data.detail[0].msg;

                break;

              default:
                alert("Something went wrong");
            }
          })
    }
    return {
      submit,
      formError,
      isValidForm,
      isValidFirstname,
      isValidLastname,
      isValidEmail,
      isValidPassword,
    }
  },
}
</script>

<style scoped>

.form-floating, .btn {
  margin-top: 10px;
}

</style>