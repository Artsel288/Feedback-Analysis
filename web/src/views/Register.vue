<template>
  <main>
    <form class="form mx-auto" @submit.prevent="submit">

      <ul class="form-selector nav nav-tabs justify-content-center">
        <li class="nav-item">
          <button :class="{ active: activeTab === 'teacher' }"
                  @click="activeTab = 'teacher'" class="nav-link" type="button" aria-current="page">Teacher</button>
        </li>
        <li class="nav-item">
          <button :class="{ active: activeTab === 'methodist' }"
                  @click="activeTab = 'methodist'" class="nav-link" type="button" aria-current="page">Methodist</button>
        </li>
      </ul>

      <div v-show="activeTab === 'teacher'">
      <RegisterTeacherForm />
      </div>

      <div v-show="activeTab === 'methodist'">
        <RegisterMethodistForm />
      </div>

    </form>
  </main>
</template>

<script>
import axios from 'axios';
import {useRouter} from "vue-router";
import RegisterMethodistForm from "@/components/RegisterMethodistForm";
import RegisterTeacherForm from "@/components/RegisterTeacherForm";

export default {
  name: "Register",
  components: {RegisterTeacherForm, RegisterMethodistForm},
  setup() {
    const router = useRouter();

    const submit = async e => {
      const form = new FormData(e.target);

      const inputs = Object.fromEntries(form.entries());
      console.log(inputs)

      await axios.post('register', inputs);

      await router.push('/login');
    }

    return {
      submit,
    }
  },

  data() {
    return {
      activeTab: 'teacher',
    }
  },
}
</script>

<style scoped>

.form{
  width: 100%;
  max-width: 400px;
}

.form-selector{
  margin: 20px;
  height: 100%;
}

</style>