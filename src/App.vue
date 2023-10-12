<template>
  <div class="flex flex-row">
    <NavBar/>
    <router-view/>
  </div>
</template>

<script>
import NavBar from "./components/NavBar.vue"
export default {
  components: { NavBar }
}
</script>

<style>

</style>