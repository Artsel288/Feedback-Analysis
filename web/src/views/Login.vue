<template>
  <main>
    <div class="form mx-auto">

      <ul class="form-selector nav nav-tabs justify-content-center">
        <li class="nav-item">
          <button :class="{ active: activeTab === 'teacher' }"
                  @click="activeTab = 'teacher'" class="nav-link" type="button" aria-current="page">Teacher</button>
        </li>
        <li class="nav-item">
          <button :class="{ active: activeTab === 'methodist' }"
                  @click="activeTab = 'methodist'" class="nav-link" type="button" aria-current="page">Methodist</button>
        </li>
        <li class="nav-item">
          <button :class="{ active: activeTab === 'admin' }"
                  @click="activeTab = 'admin'" class="nav-link" type="button" aria-current="page">Admin</button>
        </li>
      </ul>

      <div v-if="activeTab === 'teacher'">
        <LoginTeacherForm />
      </div>

      <div v-if="activeTab === 'methodist'">
        <LoginMethodistForm />
      </div>

      <div v-if="activeTab === 'admin'">
        <LoginAdminForm />
      </div>

    </div>
  </main>
</template>

<script>
import LoginTeacherForm from "@/components/LoginTeacherForm";
import LoginMethodistForm from "@/components/LoginMethodistForm";
import LoginAdminForm from "@/components/LoginAdminForm";

export default {
  name: "Login",
  components: {LoginAdminForm, LoginMethodistForm, LoginTeacherForm},
  data() {
    return {
      activeTab: 'teacher'
    }
  }
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

.form-selector a{
  cursor: pointer;
}

</style>
